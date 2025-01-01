package com.david;

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

import static org.apache.spark.sql.functions.col;
import static org.apache.spark.sql.functions.when;

public class MachineLearning {

    public static void classifyCityBySalaryAndAge(Dataset<Row> df) {
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

    public static void binaryClassificationOfSalaryLevels(Dataset<Row> df) {
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

    public static void predictSalaryByAge(Dataset<Row> df) {

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

}
