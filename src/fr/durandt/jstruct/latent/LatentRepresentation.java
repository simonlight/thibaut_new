package fr.durandt.jstruct.latent;


public class LatentRepresentation<X,H> {

	public X x;
	public H h;
	
	public LatentRepresentation(X x, H h) {
		this.x = x;
		this.h = h;
	}
}
