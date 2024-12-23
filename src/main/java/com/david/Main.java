package com.david;

import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.sql.SparkSession;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Main {
    public static void main(String[] args) {
        try (final SparkSession sparkSession = SparkSession.builder().appName("Spark Application").master("local").getOrCreate();
             final JavaSparkContext sparkContext = new JavaSparkContext(sparkSession.sparkContext())) {

            List<Integer> list = Stream.iterate(0, i -> i + 1).limit(10).collect(Collectors.toUnmodifiableList());

            JavaRDD<Integer> parallelize = sparkContext.parallelize(list).filter(e -> e % 2 == 0);
            System.out.println("Count of elements: " + parallelize.count());
        }
    }
}