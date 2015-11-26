/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.ranking;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LatentCouple<H> {
	
	private H hi = null;
	private H hj = null;
	
	public LatentCouple() {
		super();
	}
	
	/**
	 * @param hi
	 * @param hj
	 */
	public LatentCouple(H hi, H hj) {
		super();
		this.hi = hi;
		this.hj = hj;
	}
	
	/**
	 * @return the hi
	 */
	public H getHi() {
		return hi;
	}
	/**
	 * @param hi the hi to set
	 */
	public void setHi(H hi) {
		this.hi = hi;
	}
	/**
	 * @return the hj
	 */
	public H getHj() {
		return hj;
	}
	/**
	 * @param hj the hj to set
	 */
	public void setHj(H hj) {
		this.hj = hj;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		return "LatentCouple [hi=" + hi + ", hj=" + hj + "]";
	}
	
}
