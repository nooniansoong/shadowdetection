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
 * @author marko
 */
public class BinaryToText {
    
    /**
     * converts file with chars to file with string which represents char values
     * @param input
     * @param output
     * @throws Exception 
     */
    public static void convert(String input, String output) throws Exception{                
        BufferedReader reader = new BufferedReader(new FileReader(input));
        BufferedWriter writer = new BufferedWriter(new FileWriter(output));
        String line = reader.readLine();
        while (line != null){
            int val = line.charAt(0);
            String valStr = Integer.toString(val);
            writer.write(valStr);
            line = reader.readLine();
            if (line != null)
                writer.newLine();
        }
        writer.flush();
        writer.close();
        reader.close();
    }
    
}
