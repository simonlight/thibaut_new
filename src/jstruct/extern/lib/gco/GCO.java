/**
 * 
 */
package jstruct.extern.lib.gco;

import java.lang.annotation.Native;


/**
 * @author Thibaut Durand - durand.tibo@gmail.com
 *
 */
public class GCO {

	public static interface GCOLib extends Library {
		/**
		 * 
		 * @param num_pixels the number of pixels in the graph
		 * @param num_labels number of labels
		 * @param verbose
		 */
		void DoubleGeneralGraph(int num_pixels, int num_labels, double[] data, double[] smooth, int[] neighbors, int[] result, int optim, int verbose);
	}

	protected GCOLib gcolib = null;
	protected String pathLib = "/Users/thibautdurand/Desktop/These/code_c/gco_v2/Debug/libgco_v2.dylib";

	public GCO() {
		long startTime = System.currentTimeMillis();
		gcolib = (GCOLib) Native.loadLibrary(pathLib, GCOLib.class);
		long endTime = System.currentTimeMillis();
		System.out.println("load lib - Time= " + (endTime-startTime) + " ms\t" + pathLib);
	}

	/**
	 * @return the gcolib
	 */
	public GCOLib getGcolib() {
		return gcolib;
	}

	/**
	 * @param gcolib the gcolib to set
	 */
	public void setGcolib(GCOLib gcolib) {
		this.gcolib = gcolib;
	}

	/**
	 * @return the pathLib
	 */
	public String getPathLib() {
		return pathLib;
	}

	/**
	 * @param pathLib the pathLib to set
	 */
	public void setPathLib(String pathLib) {
		this.pathLib = pathLib;
	}

	
}
