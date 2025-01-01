package com.david;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class Main {

    public static final String INPUT_CSV_FILE_PATH = "src/main/resources/employees.csv";
    public static final String OUTPUT_CSV_FILE_PATH = "src/main/resources/result.csv";

    public static void main(String[] args) {
        try (final SparkSession sparkSession = SparkSession.builder().appName("Spark Application").master("local").getOrCreate();
             final JavaSparkContext sparkContext = new JavaSparkContext(sparkSession.sparkContext())) {

            //Default data set
            Dataset<Row> df = sparkSession.read().format("csv")
                    .option("header", "true")
                    .option("inferSchema", "true")
                    .load(INPUT_CSV_FILE_PATH);

            //Call the static methods of each class

            sparkContext.stop();
        }
    }


}