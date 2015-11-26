package jstruct.io;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.List;
import java.util.StringTokenizer;

import fr.lip6.jkernelmachines.type.TrainingSample;

public class FileWriterTxt {
	
	static public void writeSignatureTxt(List<double[]> l, String file){
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
			
			for(int i=0; i<l.size(); i++){
	        	for(int j=0; j<l.get(0).length; j++){
	        		bw.write(Double.toString(l.get(i)[j]) + "\t");
	        		//System.out.println(l.get(i)[j]);
	        	}
	        	bw.write("\n");
	        }
			
			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}
	
	static public void writeNameImage(List<TrainingSample<String>> l, String file){
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
			
			for(int i=0; i<l.size(); i++){
	        	bw.write(l.get(i).sample + "\n");
	        }
			
			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}
	
	static public void writeTrainingFile(List<TrainingSample<String>> l, String file){
		
		File f = new File(file);
		f.getParentFile().mkdirs();
		
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
			
			bw.write(l.size() + "\n");
			for(int i=0; i<l.size(); i++){
	        	bw.write(l.get(i).sample + "\t" + l.get(i).label + "\n");
	        }
			
			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			e.printStackTrace();
			return;
		}
	}
	
	static public void writeTrainingFile(List<TrainingSample<String>> l, String file, String pathIntegralImage){
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
			
			bw.write(l.size() + "\n");
			for(int i=0; i<l.size(); i++){
				StringTokenizer st = new StringTokenizer(l.get(i).sample, "/");
				String name = null;
				while (st.hasMoreTokens()) { 
					name = st.nextToken(); 
				}
				st = new StringTokenizer(name, ".");
	    		String IntegralImageOutput = pathIntegralImage + "/" + st.nextToken() +".im";
				
	        	bw.write(l.get(i).sample + "\t" + l.get(i).label + "\t" + IntegralImageOutput + "\n");
	        }
			
			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}
	
	static public void writeSignature(List<List<Double>> l, File file){
		System.out.println("write signature - " + file.getAbsolutePath() + "\tnb signatures " + l.size() + "x" + l.get(0).size());
		File f = new File(file.getParent());
		f.mkdirs();
		
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
			
			for(int i=0; i<l.size(); i++){
	        	for(int j=0; j<l.get(0).size(); j++){
	        		bw.write(Double.toString(l.get(i).get(j)) + "\t");
	        	}
	        	bw.write("\n");
	        }
			
			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}
}
