/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.electricfire.shadowdetectiontools.statistics;

import com.electricfire.shadowdetectiontools.util.FilesCollector;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author marko
 */
public class GetDataPeaks {
    class Peak{
       double coef = 0.;
       int numOfClass1Results = 0;
       int numOfClass2Results = 0;
    };
    
    class Peaks{

        public Peaks() {
            peaks = new ArrayList<>();
            numOfValues = 0;
            percentOfPeaks = 0.f;
        }
        
        
        protected List<Peak> peaks;
        protected int numOfValues;
        protected float percentOfPeaks;
        protected void calculatePercents(){
            percentOfPeaks = (float)(peaks.size() * 100) / (float)numOfValues;
        }
        protected void addPeak(Peak peak){
            peaks.add(peak);
        }
        protected void incNumOfValues(){
            numOfValues++;
        }
        
        @Override
        public String toString(){
            String retStr = "";
            retStr += numOfValues + "\t" + peaks.size() + "\t" + percentOfPeaks;
            return retStr;
        }
    };
    
    public void getPeaks(String rootDir, String prefix, int diffPercentage, 
                        String outFile) throws Exception{
        BufferedWriter writer = new BufferedWriter(new FileWriter(outFile));
        List<String> prefixes = new ArrayList<>();
        prefixes.add(prefix);
        List<File> files = FilesCollector.getAllFiles(rootDir, prefixes, null, false);
        for (File file : files) {
            Peaks peaks = getPeaksForArgument(diffPercentage, file);            
            peaks.calculatePercents();
            String line = FilesCollector.getFileName(file) + "\t" + peaks.toString();
            writer.write(line);
            writer.newLine();
        }
        writer.flush();
        writer.close();
    }
    
    private Peaks getPeaksForArgument(int diffPercentage, File file) throws Exception{
        Peaks peaks = new Peaks();
        BufferedReader reader = new BufferedReader(new FileReader(file));
        String line = reader.readLine();
        //skip first line
        line = reader.readLine();
        while (line != null){
            String[] tokens = line.split("\t");
            if (tokens.length > 2){
                double coef = Double.parseDouble(tokens[0]);
                int val1 = Integer.parseInt(tokens[1]);
                int val2 = Integer.parseInt(tokens[2]);
                int max = Math.max(val2, val1);
                int delta = Math.abs(val2 - val1);
                int wantedDelta = (int)(((float)max / 100.f) * (float)diffPercentage);
                if (delta >= wantedDelta){
                    Peak peak = new Peak();
                    peak.coef = coef;
                    peak.numOfClass1Results = val1;
                    peak.numOfClass2Results = val2;
                    peaks.addPeak(peak);
                }
            }
            peaks.incNumOfValues();
            line = reader.readLine();
        }
        reader.close();
        return peaks;
    }
    
}
