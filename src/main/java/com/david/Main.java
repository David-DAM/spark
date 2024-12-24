package com.david;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.apache.spark.sql.functions.max;
import static org.apache.spark.sql.functions.min;

public class Main {
    public static void main(String[] args) {
        try (final SparkSession sparkSession = SparkSession.builder().appName("Spark Application").master("local").getOrCreate();
             final JavaSparkContext sparkContext = new JavaSparkContext(sparkSession.sparkContext())) {

            String inputCsvFilePath = "src/main/resources/employees.csv";
            String outputCsvFilePath = "src/main/resources/result.csv";

            Dataset<Row> df = sparkSession.read().format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load(inputCsvFilePath);

            Dataset<Row> rowDataset = maxAndMin(df);
            saveData(rowDataset, outputCsvFilePath);
            sparkContext.stop();
        }
    }

    private static void saveData(Dataset<Row> rowDataset, String outputCsvFilePath) {
        rowDataset.write()
                .format("csv")
                .option("header", "true")
                .mode("overwrite")
                .save(outputCsvFilePath);
    }

    private static Dataset<Row> maxAndMin(Dataset<Row> df) {
        return df.agg(max("salary"), min("salary"));
    }

    private static void parallelizeCollection(JavaSparkContext sparkContext) {
        List<Integer> list = Stream.iterate(0, i -> i + 1).limit(10).collect(Collectors.toUnmodifiableList());

        JavaRDD<Integer> parallelize = sparkContext.parallelize(list).filter(e -> e % 2 == 0);
        System.out.println("Count of elements: " + parallelize.count());
    }

    private static Dataset<Row> datasetGroupBy(Dataset<Row> df) {

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
}