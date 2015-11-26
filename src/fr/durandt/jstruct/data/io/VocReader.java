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
public class VocReader {


	public static void main(String[] args) {
		String filename = "/Volumes/Eclipse/LIP6/simulation/VOC2011_Action/cvpr2014/VOCdevkit/VOC2011/ImageSets/Action/jumping_test.txt";
		List<VocData> data = readActionTest(new File(filename));
		
		for(int i=0; i<5; i++) {
			data.get(i).print();
		}
	}

	public static List<VocData> readActionTest(File file) {

		List<VocData> list = null;

		if(file.exists()) {
			try {
				System.out.println("read file: " + file.getAbsolutePath());
				InputStream ips = new FileInputStream(file); 
				InputStreamReader ipsr = new InputStreamReader(ips);
				BufferedReader br = new BufferedReader(ipsr);

				list = new ArrayList<VocData>();

				String ligne;
				while((ligne=br.readLine()) != null) {
					VocData data = new VocData();

					StringTokenizer st = new StringTokenizer(ligne);
					data.setName(st.nextToken());
					data.setIndexRegionAction(Integer.parseInt(st.nextToken()));
					
					list.add(data);
					
				}

				br.close();
			}
			catch (IOException e) {
				System.out.println("Error parsing file " + file);
				return null;
			}
			
			System.out.println("number of examples= " + list.size());
		}
		else {
			System.out.println("file " + file.getAbsolutePath() + " does not exist");
		}

		return list;
	}
}