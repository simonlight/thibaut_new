/**
 * 
 */
package fr.durandt.jstruct.latent.lssvm.segmentation;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import fr.durandt.jstruct.latent.LatentRepresentation;
import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.variable.BagImageSeg;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class SegmentationLSSVMPegasosBagImageSeg extends SegmentationLSSVMPegasos<BagImageSeg> {

	/**
	 * 
	 */
	private static final long serialVersionUID = 8559457332078833706L;

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.segmentation.SegmentationLSSVMPegasos#phi(java.lang.Object, int)
	 */
	@Override
	protected double[] phi(BagImageSeg x, int sp) {
		return x.getInstance(sp);
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.segmentation.SegmentationLSSVMPegasos#phi(java.lang.Object)
	 */
	@Override
	protected double[] phi(BagImageSeg x) {
		return x.getBagFeature();
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.segmentation.SegmentationLSSVMPegasos#getNumberOfSuperpixels(java.lang.Object)
	 */
	@Override
	protected int getNumberOfSuperpixels(BagImageSeg x) {
		return x.numberOfInstances();
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.latent.lssvm.LSSVM#init(java.util.List)
	 */
	@Override
	protected void init(List<STrainingSample<LatentRepresentation<BagImageSeg, Integer[]>, Integer[]>> l) {

		int nbClass = l.get(0).output.length;

		listClass = new ArrayList<Integer>();
		for(int i=0; i<nbClass; i++) {
			listClass.add(i);
		}

		// Compute the dimension of w
		dim = 2*listClass.size()*l.get(0).input.x.getInstance(0).length;
		
		offsetY = listClass.size()*l.get(0).input.x.getInstance(0).length;

		// Initialize w
		w = new double[dim];

		System.out.println("Segmentation Latent SSVM - classes: " + listClass);

	}
	
	public double evaluationPerClass(List<STrainingSample<LatentRepresentation<BagImageSeg, Integer[]>, Integer[]>> l) {

		// Compute the confusion matrix
		long[][] confusion = new long[listClass.size()][listClass.size()];
		for(STrainingSample<LatentRepresentation<BagImageSeg, Integer[]>, Integer[]> ts : l) {
			Object[] res = predictionOutputLatent(ts.input.x, w);
			Integer[] yp = (Integer[]) res[1]; 
			BufferedImage predictedMask = ts.input.x.predictedMask(yp);
			BufferedImage maskGT = ts.input.x.getGtMask();
			for(int i=0; i<predictedMask.getHeight(); i++) {
				for(int j=0; j<predictedMask.getWidth(); j++) {
					int pixel = predictedMask.getRGB(j, i);
					int predictLabel = (pixel) & 0xff;
					pixel = maskGT.getRGB(j, i);
					int gtLabel = (pixel) & 0xff;
					if(gtLabel != 0) {
						confusion[gtLabel-1][predictLabel-1]++;
					}
				}
			}	
		}

		// Compute G
		long[] g = new long[listClass.size()];
		for(int i=0; i<listClass.size(); i++) {
			for(int j=0; j<listClass.size(); j++) {
				g[i] += confusion[i][j];
			}
		}

		// Compute the per-class accuracy
		double pc = 0;
		for(int i=0; i<listClass.size(); i++) {
			if(g[i] != 0) {
				pc += (double)confusion[i][i] / (double)g[i];
			}
		}
		pc /= listClass.size();

		System.out.println("per class accuracy= " + pc);
		return pc;
	}

}
