/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.electricfire.shadowdetectiontools.statistics;

import com.electricfire.shadowdetectiontools.util.Pair;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 *
 * @author marko
 */
public class AnalyzeTrainingData {    
    
    private final List< Map< Double, Pair<Integer, Integer> > > resultList;
    
    public AnalyzeTrainingData(){
        resultList = new ArrayList<>();
    }
    
    public void analyze(String inputFile, String outputFile) throws Exception{
        getResults(inputFile);
        saveResults(outputFile);
    }
    
    protected void getResults(String dataFile) throws Exception{
        BufferedReader reader = new BufferedReader(new FileReader(dataFile));
        String line = reader.readLine();
        while (line != null){
            processLine(line);
            line = reader.readLine();
        }
        reader.close();
    }
    
    protected void saveResults(String outputFile) throws Exception{        
        for (int i = 0; i < resultList.size(); i++){
            String file = outputFile + (i + 1);
            BufferedWriter writer = new BufferedWriter(new FileWriter(file));
            Map< Double, Pair<Integer, Integer> > mappedValues = resultList.get(i);
            Set<Double> values = mappedValues.keySet();
            Iterator<Double> iter = values.iterator();
            String line = "Number\tWithDetected\tWithNotDetected";
            writer.write(line);
            line = "";
            while (iter.hasNext()){
                double val = iter.next();
                Pair<Integer, Integer> counters = mappedValues.get(val);
                line = val + "\t" + counters.e1 + "\t" + counters.e2;
                writer.newLine();
                writer.write(line);
            }
            writer.flush();
            writer.close();
        }        
    }
    
    protected void processLine(String line){
        String[] tokens = line.split("\t");
        if (tokens.length > 1){
            int result = Integer.parseInt(tokens[0]);
            for (int i = 1; i < tokens.length; i++){
                double val = Double.parseDouble(tokens[i]);
                processArgument(val, result, i - 1);
            }
        }
    }
    
    protected void processArgument(double argVal, int result, int index){
        Map< Double, Pair<Integer, Integer> > mappedValues = null;
        if (resultList.size() < index + 1){
            mappedValues = new HashMap<>();
            resultList.add(index, mappedValues);
        }
        else{
            mappedValues = resultList.get(index);
        }
        Pair<Integer, Integer> counters = mappedValues.get(argVal);
        if (counters == null){
            counters = new Pair<>();
            counters.e1 = 0;
            counters.e2 = 0;
            mappedValues.put(argVal, counters);
        }
        if (result == 0)
            counters.e1 = counters.e1 + 1;
        else
            counters.e2 = counters.e2 + 1;
    }
    
}
