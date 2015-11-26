package fr.durandt.jstruct.solver;

import java.util.List;

public class MosekSolver {
	
	/**
	 * Solve QP with 1 thread Mosek solver 
	 * @param gram
	 * @param lc
	 * @param c
	 * @return the solution
	 */
	public static double[] solveQP(double[][] gram, List<Double> lc, double c) {
		return solveQP(gram, lc, c, 1);
	}

	/**
	 * Solve QP with multi-threads Mosek solver
	 * @param gram
	 * @param lc
	 * @param c
	 * @param numThreads number of threads (0 = use all available cores)
	 * @return the solution
	 */
	public static double[] solveQP(double[][] gram, List<Double> lc, double c, int numThreads) {
		
		mosek.Env env = null;
		mosek.Task task = null;
		
		double[] alphas = null;
		
		try {
			env = new mosek.Env();
			task = new mosek.Task(env,0,0);
			
			// Set the number of threads
			task.putintparam(mosek.Env.iparam.num_threads, numThreads);
			
			task.set_Stream(mosek.Env.streamtype.log, new mosek.Stream() {public void stream(String msg) {}});
		
			int numcon = 1;
			task.appendcons(numcon); // number of constraints
			
			int numvar = lc.size();	// number of variables
			task.appendvars(numvar);	
			
			for(int i=0; i<numvar; i++) {
				// linear term in the objective
				task.putcj(i, -lc.get(i));
				
				// bounds on variable i
				task.putbound(mosek.Env.accmode.var, i, mosek.Env.boundkey.ra, 0.0, c);
				
				// input column i of A
				int[] asub = {0};
				double[] aval = {1.};
				task.putacol(i,asub,aval);
			}
			
			// bounds on constraints 
			for(int i=0; i<numcon; i++) {	
				task.putbound(mosek.Env.accmode.con, i, mosek.Env.boundkey.ra, 0., c);
			}
		
			//The lower triangular part of the Q matrix in the objective is specified. 
			int[] qi = new int[numvar*(numvar+1)/2];
			int[] qj = new int[numvar*(numvar+1)/2];
			double[] qval = new double[numvar*(numvar+1)/2];
			
			int n=0;
			for(int i=0; i<lc.size(); i++) {
				for(int j=0; j<=i; j++) {
					qi[n] = i;
					qj[n] = j;
					qval[n] = gram[i][j];
					n++;
				}
			}
			
			// Input the Q for the objective
			task.putqobj(qi, qj, qval);
			
			// Solve the problem
		    @SuppressWarnings("unused")
			mosek.Env.rescode r = task.optimize(); 
		    //System.out.println(" Mosek warning:" + r.toString());
		    
		    // Print a summary containing information about the solution for debugging purposes 
		    task.solutionsummary(mosek.Env.streamtype.msg); 
		    
		    mosek.Env.solsta solsta[] = new mosek.Env.solsta[1]; 
		    // Get status information about the solution 
		    task.getsolsta(mosek.Env.soltype.itr,solsta); 
		    
		    // Get the solution
		    alphas = new double[numvar];
		    task.getxx(mosek.Env.soltype.itr, alphas); 
		    
		    //System.out.println("MOSEK - alphas " + Arrays.toString(alphas));
		}
	    catch (mosek.Exception e) { 
	      System.out.println ("An error/warning was encountered"); 
	      System.out.println (e.toString()); 
	      throw e; 
	    } 
	    finally { 
		    if(task != null) {
		    	task.dispose(); 
		    }
		    if(env != null) {
		    	env.dispose();
		    }
	    }
		
		return alphas;
	}
	
}
