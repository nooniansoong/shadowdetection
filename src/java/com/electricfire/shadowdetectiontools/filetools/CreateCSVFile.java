/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.electricfire.shadowdetectiontools.filetools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

/**
 *
 * @author marko
 */
public class CreateCSVFile {
    
    public static void createCSV(String rootDir, String outFile) throws Exception{
        File root = new File(rootDir);
        BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
        process(root, writer);
        writer.flush();
        writer.close();
    }
    
    private static void writeFile(File file, BufferedWriter writer) throws Exception{
        String name = file.getName();
        int dotIndex = name.lastIndexOf('.');
        if (dotIndex != -1){
            String extension = name.substring(dotIndex + 1);
            if (extension.compareToIgnoreCase("jpg") == 0 || extension.compareToIgnoreCase("jpeg") == 0 ||
                extension.compareToIgnoreCase("png") == 0 || extension.compareToIgnoreCase("tif") == 0 ||
                extension.compareToIgnoreCase("tiff") == 0){
                name = name.substring(0, dotIndex);
                writer.write(file.getAbsolutePath() + "\t" + name + "Shadow." + extension);
                writer.newLine();
            }
        }
    }
    
    private static void process(File root, BufferedWriter writer) throws Exception{
        if (root.isFile()){
            writeFile(root, writer);
        }
        else{
            File[] subFiles = root.listFiles();
            for (int i = 0; i < subFiles.length; i++){
                File subFile = subFiles[i];
                if (subFile.isFile()){
                    writeFile(subFile, writer);
                }
                else{
                    process(subFile, writer);
                }
            }
        }
    }
    
}
