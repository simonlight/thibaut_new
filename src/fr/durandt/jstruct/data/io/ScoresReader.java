/**
 * 
 */
package fr.durandt.jstruct.data.io;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class ScoresReader {

	static public List<double[]> readFile(File file){

		List<double[]> l = null;
		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);

			System.out.print("Read scores: " + file);
			
			String ligne = br.readLine();
			StringTokenizer st = new StringTokenizer(ligne);
			//System.out.println(ligne);
			int nbExamples = Integer.parseInt(st.nextToken());
			int nbScores = Integer.parseInt(st.nextToken());

			System.out.println("\t nb examples: " + nbExamples + "\tnb scores: " + nbScores);
			l = new ArrayList<double[]>(nbExamples);

			for(int i=0; i<nbExamples; i++) {
				System.out.print(".");
				if(i>0 && i % 100 == 0) System.out.print(i);
				ligne=br.readLine();
				st = new StringTokenizer(ligne);
				double[] scoresBag = new double[nbScores];
				for(int j=0; j<nbScores; j++) {
					scoresBag[j] = Double.parseDouble(st.nextToken());
				}
				l.add(scoresBag);
			}
			System.out.println();

			br.close();
		}
		catch (IOException e) {
			System.out.println(e.getMessage());
			System.out.println("Error parsing file " + file);
			System.out.println("nb bags: " + l.size());
			return null;
		}
		return l;
	}

}
