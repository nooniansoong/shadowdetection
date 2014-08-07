/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.electricfire.shadowdetectiontools.filetools;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

/**
 *
 * @author marko
 */
public class RemoveRecordsFromSvmFile {
    
    public void removeEveryNth(int n, String inputFile, String outputFile) throws Exception{
        
    }
    
    public static void removeEveryNthPair(int n, String inputFile, String outputFile) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        int counter = 0;
        String line = reader.readLine();
        while (line != null){
            int index = counter / 2;
            if (index % n != 0){
                writer.write(line);
                writer.newLine();
            }            
            counter++;
            line = reader.readLine();
        }
        reader.close();
        writer.flush();
        writer.close();
    }
    
}
