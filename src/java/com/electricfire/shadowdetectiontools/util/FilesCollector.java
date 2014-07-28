/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.electricfire.shadowdetectiontools.util;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/**
 *
 * @author marko
 */
public class FilesCollector {
    
    private final List<File> collectedFiles = new ArrayList<>();
    
    public static List<File> getAllFiles(String rootPath, List<String> prefixes, 
                                            Set<String> extensions, boolean recursive){
        FilesCollector collector = new FilesCollector();
        File rootDir = new File(rootPath);
        if (rootDir.exists()){
            collector.process(rootDir, prefixes, extensions, recursive);
        }
        return collector.collectedFiles;
    }
    
    public static String getFileName(File file){
        if (file.isFile() == false){
            return null;
        }
        String name = file.getName();
        int dotIndex = name.lastIndexOf('.');
        if (dotIndex != -1){
            name = name.substring(0, dotIndex);
        }
        return name;
    }
    
    public static String getFileExtension(File file){
        if (file.isFile() == false){
            return null;
        }
        String name = file.getName();
        int dotIndex = name.lastIndexOf('.');
        if (dotIndex != -1){
            String extension = name.substring(dotIndex + 1);
            return extension;
        }
        return "";
    }
    
    private static boolean isAcceptableFile(File file, List<String> prefixes, 
                                            Set<String> extensions){
        
        String name = FilesCollector.getFileName(file);
        boolean found = false;
        if (prefixes != null){
            for (int i = 0; i < prefixes.size(); i++){
                String prefix = prefixes.get(i);
                if (name.startsWith(prefix)){
                    found = true;
                    break;
                }
            }
        }
        else{
            found = true;
        }
        if (found == false)
            return false;
        if (extensions == null)
            return found;
        String extension = FilesCollector.getFileExtension(file);
        return extensions.contains(extension);
    }
    
    private void process(File rootDir, List<String> prefixes, Set<String> extensions,
                        boolean recursive){
        if (rootDir.isFile() && isAcceptableFile(rootDir, prefixes, extensions)){
            collectedFiles.add(rootDir);
        }
        else{
            File[] subFiles = rootDir.listFiles();
            for (int i = 0; i < subFiles.length; i++){
                File subFile = subFiles[i];
                if (subFile.isFile()){
                    if (isAcceptableFile(subFile, prefixes, extensions)){                        
                        collectedFiles.add(subFile);
                    }
                }
                else if (recursive){
                    process(subFile, prefixes, extensions, recursive);
                }
            }
        }
    }
    
}
