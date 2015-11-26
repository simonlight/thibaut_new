/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.ranking;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.StringTokenizer;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class RankingAPMantraM2CccpCuttingPlane1SlackBagImageRegion extends RankingAPMantraM2CccpCuttingPlane1Slack<BagImageRegion, Integer> {


	/**
	 * 
	 */
	private static final long serialVersionUID = 2220058296374581854L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>> l) {
		// Print the classes and the number of relevant/irrelevant examples
		System.out.println("Rank AP MANTRA \t P= " + l.get(0).input.x.getNpos() + "\t N= " + l.get(0).input.x.getNneg());

		// Define the dimension of w
		dim = l.get(0).input.x.getFeature(0,0).length;

		// Initialize w
		w = new double[dim];
		w[0] = 1;

	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.mantra.iccv15.ranking.RankingMantraM2#valueOfHPlusMinus(java.lang.Object, double[])
	 */
	@Override
	protected Object[] valueOfHPlusMinus(BagImageRegion x, double[] w) {
		Integer hmax = null;
		Integer hmin = null;
		double valmax = -Double.MAX_VALUE;
		double valmin = Double.MAX_VALUE;
		for(int h=0; h<x.numberOfInstances(); h++) {
			double[] phi = x.getInstance(h);
			double val = linear.valueOf(w, phi);
			if(val>valmax){
				valmax = val;
				hmax = h;
			}
			if(val<valmin){
				valmin = val;
				hmin = h;
			}
		}
		Object[] res = new Object[4];
		res[0] = hmax;
		res[1] = valmax;
		res[2] = hmin;
		res[3] = valmin;
		return res;
	}


	public void writePrediction(File file, List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>> l) {
		file.getParentFile().mkdirs();
		System.out.println("write results " + file.getAbsolutePath());

		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			bw.write("name \t gt \t score \t hmax \n");
			STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput> ts = l.get(0);
			for(int i=0; i<ts.input.x.getNumberOfExamples(); i++){
				bw.write(ts.input.x.getExample(i).getName() + "\t" + ts.output.getLabel(i) + "\t");
				Object[] res = valueOfHPlusMinus(ts.input.x.getExample(i), w);
				Integer hmax = (Integer) res[0];
				double valmax = (Double) res[1];
				Integer hmin = (Integer) res[2];
				double valmin = (Double) res[3];
				double score = valmax + valmin;
				bw.write(score + "\t" + hmax + "\t" + ts.input.x.getExample(i).getRegion(hmax)[0]
						+ "\t" + ts.input.x.getExample(i).getRegion(hmax)[1] + "\t" + ts.input.x.getExample(i).getRegion(hmax)[2]
								+ "\t" + ts.input.x.getExample(i).getRegion(hmax)[3]);
				bw.write("\n");
			}

			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}


	public void writeScores(File file, List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>> l) {
		file.getParentFile().mkdirs();
		System.out.println("write scores " + file.getAbsolutePath());

		if(!file.exists()) {
			try {
				OutputStream ops = new FileOutputStream(file); 
				OutputStreamWriter opsr = new OutputStreamWriter(ops);
				BufferedWriter bw = new BufferedWriter(opsr);

				bw.write(l.get(0).input.x.getNumberOfExamples() + "\t1\n");
				for(int i=0; i<l.get(0).input.x.getNumberOfExamples(); i++) {
					Object[] res = valueOfHPlusMinus(l.get(0).input.x.getExample(i), w);
					double valmax = (Double) res[1];
					double valmin = (Double) res[3];
					double score = valmax + valmin;
					bw.write(score + "\n");
				}

				System.err.println("write file " + file.getAbsolutePath());
				bw.close();
			}
			catch (IOException e) {
				System.out.println("Error parsing file "+ file);
				return;
			}
		}
	}
	
	public void writeScoresMaxMin(File file, List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<LatentCoupleMinMax<Integer>>>, RankingOutput>> l) {
		file.getParentFile().mkdirs();
		System.out.println("write scores " + file.getAbsolutePath());

		if(!file.exists()) {
			try {
				OutputStream ops = new FileOutputStream(file); 
				OutputStreamWriter opsr = new OutputStreamWriter(ops);
				BufferedWriter bw = new BufferedWriter(opsr);

				bw.write(l.get(0).input.x.getNumberOfExamples() + "\t2\n");
				for(int i=0; i<l.get(0).input.x.getNumberOfExamples(); i++) {
					Object[] res = valueOfHPlusMinus(l.get(0).input.x.getExample(i), w);
					double valmax = (Double) res[1];
					double valmin = (Double) res[3];
					bw.write(valmax + "\t" + valmin + "\n");
				}

				System.err.println("write file " + file.getAbsolutePath());
				bw.close();
			}
			catch (IOException e) {
				System.out.println("Error parsing file "+ file);
				return;
			}
		}
	}
	
	public List<BagImageRegion> readPrediction(File file) {
		List<BagImageRegion> l = null;
		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);
			
			l = new ArrayList<BagImageRegion>();
			
			String ligne;
			ligne=br.readLine();
			while ((ligne=br.readLine()) != null){
				StringTokenizer st = new StringTokenizer(ligne);
				String name = st.nextToken();
				int gt = Integer.parseInt(st.nextToken());
				double score = Double.parseDouble(st.nextToken());
				int hmax = Integer.parseInt(st.nextToken());
				int bb1 = Integer.parseInt(st.nextToken());
				int bb2 = Integer.parseInt(st.nextToken());
				int bb3 = Integer.parseInt(st.nextToken());
				int bb4 = Integer.parseInt(st.nextToken());
				
				BagImageRegion bag = new BagImageRegion();
				bag.setName(name);
				Integer[] region = {bb1, bb2, bb3, bb4};
				bag.addRegion(region);
				
				l.add(bag);
			}
			
			br.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file " + file);
			return l;
		}
		System.out.println("Read file: " + file + "\t nb: " + l.size());
		return l;
	}
	
	public double valueOf(BagImageRegion x) {
		Object[] res = valueOfHPlusMinus(x, w);
		double valmax = (Double) res[1];
		double valmin = (Double) res[3];
		double score = valmax + valmin;
		return score;
	}

}
