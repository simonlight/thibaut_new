/**
 * 
 */
package fr.durandt.jstruct.latent.mantra.iccv15.ranking;

/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class LatentCoupleMinMax<H> {
	
	private H hmax = null;
	private H hmin = null;
	
	public LatentCoupleMinMax() {
		super();
	}
	
	/**
	 * @param hmax
	 * @param hmin
	 */
	public LatentCoupleMinMax(H hmax, H hmin) {
		super();
		this.hmax = hmax;
		this.hmin = hmin;
	}

	/**
	 * @return the hmax
	 */
	public H getHmax() {
		return hmax;
	}

	/**
	 * @param hmax the hmax to set
	 */
	public void setHmax(H hmax) {
		this.hmax = hmax;
	}

	/**
	 * @return the hmin
	 */
	public H getHmin() {
		return hmin;
	}

	/**
	 * @param hmin the hmin to set
	 */
	public void setHmin(H hmin) {
		this.hmin = hmin;
	}

	/* (non-Javadoc)
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		return "LatentCoupleMinMax [hmax=" + hmax + ", hmin=" + hmin + "]";
	}
	
	
}
