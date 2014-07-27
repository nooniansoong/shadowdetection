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
public class ConvertSVMToCSV {
    
    public void convert(String inputFilePath, String outputFilePath) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(inputFilePath));
        BufferedWriter writer = new BufferedWriter(new FileWriter(outputFilePath));
        String line = reader.readLine();
        boolean first = true;
        while (line != null){
            String processed = processLine(line);
            if (first == false){
                writer.newLine();
            }
            else{
                first = false;
            }
            writer.write(processed);
            line = reader.readLine();
        }
        writer.flush();
        writer.close();
        reader.close();
        reader.close();
    }
    
    private String processLine(String input){
        String[] tokens = input.split(" ");
        String out = "";
        for (int i = 0; i < tokens.length; i++){
            if (i == 0){
                out += tokens[i];
            }
            else{
                String[] smallTokens = tokens[i].split(":");
                out += "\t" + smallTokens[1];
            }
        }
        return out;
    }
    
}
