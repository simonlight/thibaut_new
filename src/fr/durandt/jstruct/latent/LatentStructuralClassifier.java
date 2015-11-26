package fr.durandt.jstruct.latent;

import fr.durandt.jstruct.struct.StructuralClassifier;

public interface LatentStructuralClassifier<X,Y,H> extends StructuralClassifier<LatentRepresentation<X, H>,Y> {
	/**
	 * Compute the output and latent prediction
	 * @param x
	 * @return
	 */
	public Object[] predictionOutputLatent(X x);
}