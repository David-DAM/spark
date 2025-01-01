package com.david;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.graphx.Edge;
import org.apache.spark.graphx.Graph;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.storage.StorageLevel;
import scala.Tuple2;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.spark.sql.functions.*;

public class Main {
    public static void main(String[] args) {
        try (final SparkSession sparkSession = SparkSession.builder().appName("Spark Application").master("local").getOrCreate();
             final JavaSparkContext sparkContext = new JavaSparkContext(sparkSession.sparkContext())) {

            String inputCsvFilePath = "src/main/resources/employees.csv";
            //String outputCsvFilePath = "src/main/resources/result.csv";

            Dataset<Row> df = sparkSession.read().format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load(inputCsvFilePath);

            sparkContext.stop();
        }
    }

    private static void classifyCityBySalaryAndAge(Dataset<Row> df) {
        StringIndexer cityIndexer = new StringIndexer()
                .setInputCol("city")
                .setOutputCol("label");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"age", "salary"})
                .setOutputCol("features");

        // Create the model of classification (Decision tree)
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features");

        // Create the pipeline
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{cityIndexer, assembler, dt});

        Dataset<Row>[] splits = df.randomSplit(new double[]{0.8, 0.2}, 1234);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train the model
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate the model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Precisión global: " + accuracy);

        // Matrix of confuse
        predictions.groupBy("label", "prediction").count().show();
    }

    private static void binaryClassificationOfSalaryLevels(Dataset<Row> df) {
        // 1. Create binary label: salary >= 100,000 -> 1, other wise -> 0
        df = df.withColumn("label", when(col("salary").geq(100000), 1).otherwise(0));

        // 2. Convert column `city` to numeric values
        StringIndexer cityIndexer = new StringIndexer()
                .setInputCol("city")
                .setOutputCol("cityIndex");

        // 3. Apply OneHotEncoding to the column `cityIndex`
        OneHotEncoder cityEncoder = new OneHotEncoder()
                .setInputCol("cityIndex")
                .setOutputCol("cityVec");

        // 4. Combine the characteristics in one unique vector
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"age", "cityVec", "salary"})
                .setOutputCol("features");

        // 5. Configure  the classification model (Logistic Regression)
        LogisticRegression lr = new LogisticRegression();

        // 6. Create a pipeline with the previous stages
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{
                cityIndexer, cityEncoder, assembler, lr
        });

        Dataset<Row>[] splits = df.randomSplit(new double[]{0.8, 0.2}, 1234);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        PipelineModel model = pipeline.fit(trainingData);

        Dataset<Row> predictions = model.transform(testData);

        // Evaluate the model using AUC (Area Under Curve)
        BinaryClassificationEvaluator evaluator = new BinaryClassificationEvaluator()
                .setLabelCol("label")
                .setRawPredictionCol("rawPrediction")
                .setMetricName("areaUnderROC");

        double auc = evaluator.evaluate(predictions);
        System.out.println("AUC: " + auc);

        predictions.select("id", "name", "salary", "label", "prediction").show();
    }

    private static void salaryDifferenceGraphWithTopConnections(Dataset<Row> df) {
        // Create alias for self-join
        Dataset<Row> df1 = df.alias("e1");
        Dataset<Row> df2 = df.alias("e2");

        // Realize self-join and calculate the difference between salaries
        Dataset<Row> relationships = df1.join(df2, df1.col("id").notEqual(df2.col("id")))
                .withColumn("salary_diff",
                        abs(col("e1.salary").minus(col("e2.salary"))).divide(col("e1.salary"))
                )
                .filter(col("salary_diff").leq(0.1))
                .select(
                        col("e1.id").alias("employee1"),
                        col("e2.id").alias("employee2"),
                        col("salary_diff")
                );

        // Converter DataFrames to RDDs for GraphX
        JavaRDD<Tuple2<Object, String>> vertex = df.javaRDD().map(row ->
                new Tuple2<>(row.getInt(0), row.getString(1) + " (" + row.getString(3) + ")")
        );

        JavaRDD<Edge<String>> edges = relationships.javaRDD().map(row ->
                new Edge<>(
                        row.getInt(0), // employee1 ID
                        row.getInt(1), // employee2 ID
                        "similar_salary" // relation
                )
        );

        // Build the graph
        Graph<String, String> graph = Graph.apply(
                vertex.rdd(),
                edges.rdd(),
                "default",
                StorageLevel.MEMORY_AND_DISK(),
                StorageLevel.MEMORY_AND_DISK(),
                scala.reflect.ClassTag$.MODULE$.apply(String.class), // Class for vertex
                scala.reflect.ClassTag$.MODULE$.apply(String.class)  // Class for edges
        );

        // Analysis of the graph
        // 1. Number of de vertex y edges
        //System.out.println("Number of vertex: " + graph.vertices().count());
        //System.out.println("Number of edges: " + graph.edges().count());

        // 2. Employees with more connexions
        System.out.println("Top employees with more connections:");
        JavaPairRDD<Long, Integer> degrees = graph.edges()
                .toJavaRDD()
                .flatMapToPair(edge -> {
                    // Emit the vertex of origin and destiny for counting the grade of input and output
                    return List.of(new Tuple2<>(edge.srcId(), 1), new Tuple2<>(edge.dstId(), 1)).iterator();
                })
                .reduceByKey(Integer::sum);// Reduce by summing the connections for every vertex


        // Order by the grade (value) from higher to lower
        JavaPairRDD<Long, Integer> sortedDegrees = degrees
                .mapToPair(tuple -> new Tuple2<>(tuple._2, tuple._1)) // Change key and value to order by value
                .sortByKey(false) // Order descending by the grade
                .mapToPair(tuple -> new Tuple2<>(tuple._2, tuple._1)); // Change to the original format again

        // Take the 5 highest vertex
        sortedDegrees.take(5).forEach(degree ->
                System.out.println("Employee ID: " + degree._1 + ", Connections: " + degree._2)
        );
    }

    private static void predictSalaryByAge(Dataset<Row> df) {

        Dataset<Row> selectedData = df.select("age", "salary");

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{"age"})
                .setOutputCol("features");

        Dataset<Row> preparedData = assembler.transform(selectedData)
                .select("features", "salary");

        Dataset<Row>[] splits = preparedData.randomSplit(new double[]{0.8, 0.2}, 1234);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        LinearRegression lr = new LinearRegression()
                .setLabelCol("salary")
                .setFeaturesCol("features");

        LinearRegressionModel model = lr.fit(trainingData);

        Dataset<Row> predictions = model.transform(testData);
        predictions.show();

        System.out.println("Coefficients: " + model.coefficients());
        System.out.println("Intercept: " + model.intercept());
        System.out.println("RMSE: " + model.summary().rootMeanSquaredError());
        System.out.println("R2: " + model.summary().r2());
    }

    private static Dataset<Row> countEveryLetter(Dataset<Row> df) {
        return df.select(explode(split(df.col("city"), "")).as("letter"))
                .groupBy("letter")
                .count()
                .where("letter != ''")
                .orderBy(desc("count"));
    }

    private static Dataset<Row> averageSalary(Dataset<Row> df) {
        return df.select(format_number(avg("salary"), 2).as("Average salary"));
    }

    private static Dataset<Row> countDistinctCities(Dataset<Row> df) {
        return df.select(count_distinct(df.col("city")).as("Distinct city count"));
    }

    private static Dataset<Row> youngPeopleWithHighSalary(Dataset<Row> df) {
        return df.where("salary > 100000")
                .where("age < 30");
    }

    private static void saveData(Dataset<Row> rowDataset, String outputCsvFilePath) {
        rowDataset.write()
                .format("csv")
                .option("header", "true")
                .mode("overwrite")
                .save(outputCsvFilePath);
    }

    private static Dataset<Row> maxAndMinSalary(Dataset<Row> df) {
        return df.agg(max("salary"), min("salary"));
    }

    private static Dataset<Row> maxSalaryByCity(Dataset<Row> df) {

        return df.groupBy("city").max("salary");
    }

    private static void rddGroupBy(JavaSparkContext sparkContext, String csvFilePath) {
        JavaRDD<Employee> employeeJavaRDD = sparkContext.textFile(csvFilePath).map(line -> {
            String[] fields = line.split(",");
            Employee employee = new Employee();
            employee.setId(Long.valueOf(fields[0]));
            employee.setName(fields[1]);
            employee.setAge(Integer.parseInt(fields[2]));
            employee.setCity(fields[3]);
            employee.setSalary(Double.parseDouble(fields[4]));
            return employee;
        });

        JavaPairRDD<String, Iterable<Employee>> employeeByCities = employeeJavaRDD.groupBy(Employee::getCity);

        employeeByCities.foreach(tuple2 -> {
            System.out.println("City: " + tuple2._1);
            tuple2._2().forEach(System.out::println);
        });
    }

    private static void parallelizeCollection(JavaSparkContext sparkContext) {
        List<Integer> list = Stream.iterate(0, i -> i + 1).limit(10).collect(Collectors.toUnmodifiableList());

        JavaRDD<Integer> parallelize = sparkContext.parallelize(list).filter(e -> e % 2 == 0);
        System.out.println("Count of elements: " + parallelize.count());
    }
}