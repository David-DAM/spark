package com.david;

import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

public class Storage {

    public static void saveData(Dataset<Row> rowDataset, String outputCsvFilePath) {
        rowDataset.write()
                .format("csv")
                .option("header", "true")
                .mode("overwrite")
                .save(outputCsvFilePath);
    }

}
