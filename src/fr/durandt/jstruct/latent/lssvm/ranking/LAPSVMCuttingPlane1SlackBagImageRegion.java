/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.ranking;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput;
import fr.durandt.jstruct.ssvm.ranking.RankingOutput;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageRegion;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LAPSVMCuttingPlane1SlackBagImageRegion extends LAPSVMCuttingPlane1Slack<BagImageRegion,Integer> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 7525274829822819325L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.ranking.LAPSVMCuttingPlane1Slack#latentPrediction(java.lang.Object, double[])
	 */
	@Override
	protected Integer latentPrediction(BagImageRegion x, double[] w) {
		double max = -Double.MAX_VALUE;
		int hpredict = -1; // Latent prediction
		for(int h=0; h<x.numberOfInstances(); h++) {	// For each region
			// Compute the score of region h
			double score = linear.valueOf(w, x.getInstance(h));
			if(score > max) {
				max = score;
				hpredict = h;
			}
		}
		return hpredict;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>> l) {

		// Print the classes and the number of relevant/irrelevant examples
		System.out.println("Rank AP SSVM \t P= " + l.get(0).input.x.getNpos() + "\t N= " + l.get(0).input.x.getNneg());

		// Define the dimension of w
		dim = l.get(0).input.x.getFeature(0,0).length;

		// Initialize w
		w = new double[dim];
		w[0] = 1;

	}

	public double valueOf(BagImageRegion x) {
		Integer hPredict = latentPrediction(x);
		double score = linear.valueOf(w, x.getInstance(hPredict));
		return score;
	}

	public void writePrediction(File file, List<STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput>> l) {
		file.getParentFile().mkdirs();
		System.out.println("write results " + file.getAbsolutePath());

		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);

			bw.write("name \t gt \t score \t hmax \n");
			STrainingSample<LatentRepresentation<LatentRankingInput<BagImageRegion, Integer>, List<Integer>>, RankingOutput> ts = l.get(0);
			for(int i=0; i<ts.input.x.getNumberOfExamples(); i++){
				bw.write(ts.input.x.getExample(i).getName() + "\t" + ts.output.getLabel(i) + "\t");
				Integer hmax = latentPrediction(ts.input.x.getExample(i),w);
				double score = linear.valueOf(w, ts.input.x.getFeature(i,hmax));
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
}
