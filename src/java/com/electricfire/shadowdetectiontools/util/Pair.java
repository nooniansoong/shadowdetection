/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.electricfire.shadowdetectiontools.util;

import java.util.Objects;

/**
 *
 * @author marko
 * @param <T1>
 * @param <T2>
 */
public class Pair<T1, T2> {
    public T1 e1;
    public T2 e2;

    public Pair() {
        e1 = null;
        e2 = null;
    }
    
    @Override
    public boolean equals(Object other) {
        if (!(other instanceof Pair)) {
            return false;
        }
        Pair<?, ?> p = (Pair<?, ?>)other;
        return e1.equals(p.e1) && e2.equals(p.e2);
    }

    @Override
    public int hashCode() {
        int hash = 7;
        hash = 97 * hash + Objects.hashCode(this.e1);
        hash = 97 * hash + Objects.hashCode(this.e2);
        return hash;
    }
    
}