package fr.durandt.jstruct.latent.mantra.cvpr15.multiclass;

import java.util.ArrayList;
import java.util.List;

/**
 * Class used to store min and max values
 * 
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 * @param <H>
 */
public class ComputedScoresMinMax<H> {
	
	// prediction
	private List<H> lhmax = null;	// latent variable max
	private List<H> lhmin = null;	// latent variable min
	
	// value
	private List<Double> lvmax = null;	// latent values max
	private List<Double> lvmin = null;	// latent values min
	
	public ComputedScoresMinMax() {
		lhmax = new ArrayList<H>();
		lvmax = new ArrayList<Double>();
		
		lhmin = new ArrayList<H>();
		lvmin = new ArrayList<Double>();
	}
	
	public void add(H hmax, double vmax, H hmin, double vmin) {
		lhmax.add(hmax);
		lvmax.add(vmax);
		
		lhmin.add(hmin);
		lvmin.add(vmin);
	}
	
	public void set(int y, H hmax, double vmax, H hmin, double vmin) {
		lhmax.set(y,hmax);
		lvmax.set(y,vmax);
		
		lhmin.set(y,hmin);
		lvmin.set(y,vmin);
	}
	
	public H getHmax(int y) {
		return lhmax.get(y);
	}
	public double getVmax(int y) {
		return lvmax.get(y);
	}
	public H getHmin(int y) {
		return lhmin.get(y);
	}
	public double getVmin(int y) {
		return lvmin.get(y);
	}
	
	public int getMaxY(int ybar) {
		// ymax = max_(y!=ybar) max_h <w,Psi(x,y,h)>
		double max = -Double.MAX_VALUE;
		int ymax = -1;
		for(int y=0; y<lvmax.size(); y++) {
			if(y != ybar && lvmax.get(y)>max) {
				max = lvmax.get(y);
				ymax = y;
			}
		}
		return ymax;
	}
	
	public void setVmax(int y, double v) {
		lvmax.set(y, v);
	}
	public List<H> getLhmax() {
		return lhmax;
	}
	public void setLhmax(List<H> lhmax) {
		this.lhmax = lhmax;
	}
	public List<Double> getLvmax() {
		return lvmax;
	}
	public void setLvmax(List<Double> lvmax) {
		this.lvmax = lvmax;
	}
	public List<H> getLhmin() {
		return lhmin;
	}
	public void setLhmin(List<H> lhmin) {
		this.lhmin = lhmin;
	}
	public List<Double> getLvmin() {
		return lvmin;
	}
	public void setLvmin(List<Double> lvmin) {
		this.lvmin = lvmin;
	}	
	
	@Override
	public String toString() {
		String s = "lhmax= " + lhmax + "\nlhmin= " + lhmin + "\nlvmax= " + lvmax + "\nlvmin= " + lvmin;
		return s;
	}
}
