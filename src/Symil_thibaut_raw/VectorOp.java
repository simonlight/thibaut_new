package Symil_thibaut_raw;


import java.util.List;

import fr.lip6.jkernelmachines.util.algebra.VectorOperations;


public class VectorOp {

	public static double dot(double[] vector1, double[] vector2){
		if(vector1.length != vector2.length){
			System.out.println("ERROR dot: vector1 :" + vector1.length + "\t vector2: " + vector2.length);
			System.exit(0);
		}

		double s = 0;
		for(int i=0; i<vector1.length; i++){
			s += vector1[i]*vector2[i];
		}
		return s;
	}

	public static double getNormL1(double[] vector){
		double s = 0;
		for(int i=0; i<vector.length; i++){
			s += Math.abs(vector[i]);
		}
		return s;
	}

	public static void normL1(double[] vector){
		double norm = getNormL1(vector);
		if(norm != 0){
			for(int i=0; i<vector.length; i++){
				vector[i] = (vector[i] / norm);
			}
		}
	}

	public static double getNormL2(double[] vector){
		double s = 0;
		for(int i=0; i<vector.length; i++){
			s += vector[i]*vector[i];
		}
		return Math.sqrt(s);
	}

	public static void normL2(double[] vector){
		double norm = getNormL2(vector);
		if(norm != 0){
			for(int i=0; i<vector.length; i++){
				vector[i] = (vector[i] / norm);
			}
		}
	}

	public static double getNormL2(double[][] vector){
		double s = 0;
		for(int i=0; i<vector.length; i++){
			for(int j=0; j<vector[i].length; j++){
				s += vector[i][j]*vector[i][j];
			}
		}
		return Math.sqrt(s);
	}

	public static void normL2(double[][] vector){
		double norm = getNormL2(vector);
		if(norm != 0){
			for(int i=0; i<vector.length; i++){
				for(int j=0; j<vector[i].length; j++){
					vector[i][j] = (vector[i][j] / norm);
				}
			}
		}
	}

	public static void normL2Block(double[] vector, int n, int d){
		for(int i=0; i<n; i++) {
			double s = 0;
			for(int j=0; j<d; j++) {
				s += vector[i*d+j]*vector[i*d+j];
			}
			double norm = Math.sqrt(s);
			if(norm != 0){
				for(int j=0; j<d; j++) {
					vector[i*d+j] = (vector[i*d+j] / norm);
				}
			}
		}
	}

	/**
	 * Performs a linear combination of 2 vectors and store the result in the first vector:
	 * a = a + lambda * b
	 * @param a first and output vector
	 * @param lambda weight of the second vector
	 * @param b second vector
	 */
	public static void add(double[] a, double[] b, double lambda) {
		if(a.length != b.length) {
			System.out.println("ERROR : vectors of different size " + a.length + "\t" + b.length);
		}
		for(int i=0; i<a.length; i++) {
			a[i] += lambda * b[i];
		}
	}

	/**
	 * Performs a linear combination of 2 vectors and store the result in the first vector:
	 * a = gamma * a + lambda * b
	 * @param a first and output vector
	 * @param b second vector
	 * @param gamma weight of the first vector
	 * @param lambda weight of the second vector
	 */
	public static void add(double[] a, double[] b, double gamma, double lambda) {
		if(a.length != b.length) {
			System.out.println("ERROR : vectors of different size " + a.length + "\t" + b.length);
		}
		for(int i=0; i<a.length; i++) {
			a[i] = gamma * a[i] + lambda * b[i];
		}
	}

	/**
	 * Performs a linear combination of 2 matrix and store the result in the first vector:
	 * a = gamma * a + lambda * b
	 * @param a first and output matrix
	 * @param b second matrix
	 * @param gamma weight of the first matrix
	 * @param lambda weight of the second matrix
	 */
	public static void add(double[][] a, double[][] b, double gamma, double lambda) {
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				a[i][j] = gamma * a[i][j] + lambda * b[i][j];
			}
		}
	}

	/**
	 * Performs an addition of 2 vectors and store the result in the first vector:
	 * a = a + b
	 * @param a first and output vector
	 * @param b second vector
	 */
	public static void add(double[] a, double[] b) {
		if(a.length != b.length) {
			System.out.println("ERROR : vectors of different size " + a.length + "\t" + b.length);
		}
		for(int i=0; i<a.length; i++) {
			a[i] += b[i];
		}
	}

	/**
	 * Performs an subtraction of 2 vectors and store the result in the first vector:
	 * a = a - b
	 * @param a first and output vector
	 * @param b second vector
	 */
	public static void sub(double[] a, double[] b) {
		if(a.length != b.length) {
			System.out.println("ERROR : vectors of different size " + a.length + "\t" + b.length);
		}
		for(int i=0; i<a.length; i++) {
			a[i] -= b[i];
		}
	}

	public static double[] concatenation(double[] a, double[] b) {

		if(a == null && b == null) {
			return null;
		}
		else if(a == null && b != null) {
			//System.out.println("vector operation concatenation : a null + b: " + b.length);
			return b;
		}
		else if(a != null && b == null) {
			return a;
		}

		double[] c = new double[a.length + b.length];
		for(int i=0; i<a.length; i++) {
			c[i] = a[i];
		}
		for(int i=0; i<b.length; i++) {
			c[i+a.length] = b[i];
		}

		return c;
	}

	public static double[] addValeur(double[] vector, double valeur) {
		double[] newVector = new double[vector.length + 1];
		for(int i=0; i<vector.length; i++) {
			newVector[i] = vector[i];
		}
		newVector[vector.length] = valeur;
		return newVector;
	}

	public static double distanceL2(double[] a, double[] b) {
		double d=0;
		for(int i=0; i<a.length; i++) {
			d += (a[i]-b[i])*(a[i]-b[i]);
		}
		return Math.sqrt(d);
	}

	public static double mean(double[] f) {
		double mean = 0;
		for(int i=0; i<f.length; i++) {
			mean += f[i];
		}
		mean /= f.length;
		return mean;
	}

	public static double stddev(double[] f) {
		double m = mean(f);
		double stddev = 0;
		for(double s : f) {
			stddev += (s - m)*(s - m);
		}
		stddev = Math.sqrt(stddev/f.length);
		return stddev;
	}

	public static void mul(double[] vector, double lambda) {
		for(int i=0; i<vector.length; i++){
			vector[i] = vector[i] * lambda;
		}
	}

	/**
	 * a = a * lambda
	 * @param a
	 * @param lambda
	 */
	public static void mul(double[][] a, double lambda) {
		for(int i=0; i<a.length; i++) { 
			for(int j=0; j<a[i].length; j++) {
				a[i][j] = a[i][j] * lambda;
			}
		}
	}

	public static double[][] add(double[][] a, double[][] b) {
		double[][] c = new double[a.length][a[0].length];
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				c[i][j] = a[i][j] + b[i][j];
			}
		}
		return c;
	}

	public static double[][] add(int[][] a, double[][] b) {
		double[][] c = new double[a.length][a[0].length];
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				c[i][j] = a[i][j] + b[i][j];
			}
		}
		return c;
	}

	/**
	 * Performs an addition of 2 matrix and store the result in the matrix vector:
	 * a = a + b
	 * @param a first (and output) matrix
	 * @param b second matrix
	 */
	public static void addi(double[][] a, double[][] b) {
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				a[i][j] += b[i][j];
			}
		}
	}

	/**
	 * Performs an subtraction of 2 matrix and store the result in a new matrix:
	 * c = a - b
	 * @param a first matrix
	 * @param b second matrix
	 * @return c
	 */
	public static double[][] sub(double[][] a, double[][] b) {
		double[][] c = new double[a.length][a[0].length];
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				c[i][j] = a[i][j] - b[i][j];
			}
		}
		return c;
	}

	/**
	 * Performs an subtraction of 2 matrix and store the result in the first matrix:
	 * a = a - b
	 * @param a first (and output) matrix
	 * @param b second matrix
	 */
	public static void subi(double[][] a, double[][] b) {
		for(int i=0; i<a.length; i++) {
			for(int j=0; j<a[i].length; j++) {
				a[i][j] -= b[i][j];
			}
		}
	}

	public static double min(double[] a) {
		double min = Double.MAX_VALUE;
		for(double v : a) {
			if(v<min) {
				min = v;
			}
		}
		return min;
	}

	public static double min(double[][] a) {
		double min = Double.MAX_VALUE;
		for(double[] t : a) {
			for(double v : t) {
				if(v<min) {
					min = v;
				}
			}
		}
		return min;
	}

	/**
	 * Compute the minimum of vector a
	 * @param a
	 * @return
	 */
	public static int min(int[][] a) {
		int min = Integer.MAX_VALUE;
		for(int[] t : a) {
			for(int v : t) {
				if(v<min) {
					min = v;
				}
			}
		}
		return min;
	}

	/**
	 * Compute the maximum of vector a
	 * @param a
	 * @return
	 */
	public static double max(double[] a) {
		double max = -Double.MAX_VALUE;
		for(double v : a) {
			if(v>max) {
				max = v;
			}
		}
		return max;
	}

	public static double dot(double[] a, Double[] b) {
		double s = 0;
		for(int i=0; i<a.length; i++) {
			s += a[i] * b[i];
		}
		return s;
	}

	/**
	 * Compute the dot product for each row and return the sum of each row
	 * @param a
	 * @param b
	 * @return
	 */
	public static double dot(double[][] a, double[][] b) {
		double s = 0;
		for(int i=0; i<a.length; i++) {
			s += VectorOperations.dot(a[i], b[i]);
		}
		return s;
	}

	/**
	 * Compute the p-norm of vector x
	 * @param x n-dimension vector 
	 * @param p 
	 * @return (\sum_{i=1}^n |x[i]|^p)^(1/p)
	 */
	public static double pnorm(double[] x, double p) {
		if(p>=1) {
			double np = 0.;
			for(double xi : x) {
				np += Math.pow(Math.abs(xi), p);
			}
			return Math.pow(np,1/p);
		}
		else {
			System.out.println("Error: p-nom  p>=1 and p= " + p);
			return 0;
		}
	}

	/**
	 * Compute the maximum of vector a
	 * @param a
	 * @return
	 */
	public static Integer max(Integer[] a) {
		Integer max = -Integer.MAX_VALUE;
		for(Integer v : a) {
			if(v>max) {
				max = v;
			}
		}
		return max;
	}
	
	/**
	 * x[i] = 1 / (1 + exp(a*x[i] + b))
	 * @param x
	 * @param a
	 * @param b
	 * @return
	 */
	public static void plattScaling(double[] x, double a, double b) {
		for(int i=0; i<x.length; i++) {
			x[i] = 1. / (1. + Math.exp(a*x[i] + b));
		}
	}
	
	public static void plattScaling(List<double[]> l, double a, double b) {
		for(double[] x : l) {
			plattScaling(x, a, b);
		}
	}
	
	public static double[] mean(double[] vector1, double[] vector2){
		if(vector1.length != vector2.length){
			System.out.println("ERROR dot: vector1 :" + vector1.length + "\t vector2: " + vector2.length);
			System.exit(0);
		}

		double[] mean = new double[vector1.length];
		for(int i=0; i<vector1.length; i++){
			mean[i] = (vector1[i] + vector2[i]) / 2.;
		}
		return mean;
	}
}
