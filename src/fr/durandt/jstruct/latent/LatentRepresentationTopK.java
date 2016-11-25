package fr.durandt.jstruct.latent;

import java.util.ArrayList;

public class LatentRepresentationTopK<X, H> {

	public X x;
	public ArrayList<H> hlist;
	
	public LatentRepresentationTopK(X x, ArrayList<H> hlist) {
		this.x = x;
		this.hlist = hlist;
	}
}
