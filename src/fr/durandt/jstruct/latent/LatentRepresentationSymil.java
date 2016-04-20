package fr.durandt.jstruct.latent;

public class LatentRepresentationSymil<X, Hp, Hn> {

	public X x;
	public Hp hp;
	public Hn hn;
	
	public LatentRepresentationSymil(X x, Hp hp, Hn hn) {
		this.x = x;
		this.hp = hp;
		this.hn = hn;
	}
}
