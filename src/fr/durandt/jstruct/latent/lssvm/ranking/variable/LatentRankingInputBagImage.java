/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.ranking.variable;

import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImage;


/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LatentRankingInputBagImage extends LatentRankingInput<BagImage, Integer> {

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.ranking.variable.LatentRankingInput#getFeature(int, java.lang.Object)
	 */
	@Override
	public double[] getFeature(int i, Integer h) {
		return examples.get(i).x.getInstance(h);
	}

	
	public LatentRankingInputBagImage(List<STrainingSample<BagImage, Integer>> examples) {

		this.examples = new ArrayList<LatentRepresentation<BagImage,Integer>>(examples.size());
		labels = new ArrayList<Integer>(examples.size());
		for(int i=0; i<examples.size(); i++) {
			this.examples.add(i, new LatentRepresentation<BagImage,Integer>(examples.get(i).input, 0));
			labels.add(i, examples.get(i).output);
			if(examples.get(i).output == 1) {
				npos++;
			}
			else {
				nneg++;
			}
		}
	}
}
