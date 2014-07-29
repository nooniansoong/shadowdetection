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
import java.util.Set;

/**
 *
 * @author marko
 */
public class RemoveFeaturesFromSVMFile {
    public static void removeFeatures(String inputFile, Set<Integer> featuresToRemove, 
            String outputFile) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(inputFile));
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFile));
        String line = reader.readLine();
        while (line != null){
            String[] tokens = line.split(" ");
            if (tokens.length > 1){
                line = tokens[0];
                int counter = 1;
                for (int i = 1; i < tokens.length; i++){
                    if (featuresToRemove.contains(i) == false){                        
                        line += " ";
                        String[] smallTokens = tokens[i].split(":");
                        String member = counter + ":" + smallTokens[1];
                        line += member;
                        counter++;
                    }                    
                }
                if (line.compareTo("") != 0){
                    writer.write(line);
                    writer.newLine();
                }
            }
            line = reader.readLine();
        }
        reader.close();
        writer.flush();
        writer.close();
    }
}
