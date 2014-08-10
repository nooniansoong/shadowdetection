/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.electricfire.shadowdetectiontools.filetools;

import com.electricfire.shadowdetectiontools.util.FilesCollector;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 *
 * @author marko
 */
public class CreateCSVFile {
    
    public static void createCSV(String rootDir, String outFile) throws Exception{        
        BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
        Set<String> extensions = new HashSet<>();
        extensions.add("jpg"); extensions.add("jpeg"); extensions.add("png");
        extensions.add("tif"); extensions.add("tiff"); extensions.add("JPG");
        extensions.add("JPEG");
        List<File> files = FilesCollector.getAllFiles(rootDir, null, extensions, true);
        if (files != null){
            for (File file : files) {
                writeFile(file, writer);
            }
        }
        writer.flush();
        writer.close();
    }
    
    private static void writeFile(File file, BufferedWriter writer) throws Exception{        
        String extension = FilesCollector.getFileExtension(file);
        String name = FilesCollector.getFileName(file);
        File parent = file.getParentFile();
        writer.write(file.getAbsolutePath() + "\t" + parent.getAbsolutePath() + "/" + name + "Shadow.tif");
        writer.newLine();
    }        
    
}
