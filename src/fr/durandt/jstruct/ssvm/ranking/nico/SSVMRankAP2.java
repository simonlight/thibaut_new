package fr.durandt.jstruct.ssvm.ranking.nico;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.StringTokenizer;

import fr.durandt.jstruct.struct.STrainingSample;
import fr.durandt.jstruct.struct.StructuralClassifier;
import fr.lip6.jkernelmachines.kernel.typed.DoubleLinear;
import fr.lip6.jkernelmachines.util.DebugPrinter;
import fr.lip6.jkernelmachines.util.algebra.VectorOperations;

public class SSVMRankAP2 implements StructuralClassifier<List<double[]>, List<Integer>>, Serializable {
	
	/**
	 * Learn to rank documents where the binary labels are relevant(+1) and non-relevant(-1)
	 */
	private static final long serialVersionUID = -712338181556012795L;
	
	protected int optim = 1;
	protected double lambda = 1e-4;
	protected int cpmax = 50;
	protected int cpmin = 5;
	protected double epsilon = 1e-2;
	
	// debug
	DebugPrinter debug = new DebugPrinter();
	
	//svm hyperplane
	protected double[] w = null;
	protected List<Integer> listClass = null;
	
	//linear kernel
	protected DoubleLinear linear = new DoubleLinear();
	
	protected RankingType st = null; 
	protected double[] psigt = null;

	@Override
	public void train(List<STrainingSample<List<double[]>, List<Integer>>> l) {
		if(l.isEmpty())
			return;
		
		train(l.get(0));
	}
	
	public void train(STrainingSample<List<double[]>, List<Integer>> ts) {
		
		if(st == null) {
			System.out.println("Error st == null");
		}
		
		int dim = st.psi(ts.input,ts.output).length;
		w = new double [dim]; 
		
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train SSVM Ranking \tlambda: " + lambda + "\tdim: " + w.length);
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		if(optim == 1) {
			System.out.println("optim " + optim + " - Cutting-Plane 1 Slack - Mosek");
		}
		System.out.println("----------------------------------------------------------------------------------------");
		
		w[0] = 1;
		System.out.println("init ap= " + test(ts));
		long startTime = System.currentTimeMillis();
		if(optim == 1) {
			trainCP1SlackPrimalDual(ts);
		}
		else {
			System.out.println("ERROR Optim option invalid " + optim);
			System.exit(0);
		}
		//System.out.println("obj= " + primalObj(ts,w));
		long endTime = System.currentTimeMillis();
		System.out.println("Fin optim - Time learning= "+ (endTime-startTime)/1000 + "s");
		System.out.println("final ap= " + test(ts));
		System.out.println("----------------------------------------------------------------------------------------");
	}
	
	protected void trainCP1SlackPrimalDual(STrainingSample<List<double[]>, List<Integer>> ts) {
		double c = 1/lambda;
		int t=0;
		
		psigt = st.psi(ts.input, ts.output);
		//System.out.println(Arrays.toString(psigt));
		
		List<double[]> lg 	= new ArrayList<double[]>();
		List<Double> lc 	= new ArrayList<Double>();

		Object[] or 	= cuttingPlane(ts,w);
		double[] gt 	= (double[]) or[0];
		double ct		= (Double) or[1];
		
		lg.add(gt);
		lc.add(ct);
		
		double[][] gram = null;
		double xi=0;
		
		while(t<cpmin || (t<=cpmax && VectorOperations.dot(w,gt) < ct - xi - epsilon)) {
			
			System.out.print(".");
			if(t == cpmax) {
				System.out.print(" # max iter ");
			}
			
			if(gram != null) {
				double[][] g = gram;
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<g.length; i++) {
					for(int j=0; j<g.length; j++) {
						gram[i][j] = g[i][j];
					}
				}
				for(int i=0; i<lc.size(); i++) {
					gram[lc.size()-1][i] = VectorOperations.dot(lg.get(lc.size()-1), lg.get(i));
					gram[i][lc.size()-1] = gram[lc.size()-1][i];
				}
				gram[lc.size()-1][lc.size()-1] += 1e-8;
			}
			else {
				gram = new double[lc.size()][lc.size()];
				for(int i=0; i<gram.length; i++) {
					for(int j=i; j<gram.length; j++) {
						gram[i][j] = VectorOperations.dot(lg.get(i), lg.get(j));
						gram[j][i] = gram[i][j];
						if(i==j) {
							gram[i][j] += 1e-8;
						}
					}
				}
			}
			double[] alphas = optimMosek(gram, lg, lc, c);
			//System.out.println("alphas " + Arrays.toString(alphas));
			xi = (dot(alphas,lc.toArray(new Double[lc.size()])) - 0.5 * matrixProduct(alphas,gram))/c;
			
			// new w
			w = new double[lg.get(0).length];
			for(int i=0; i<alphas.length; i++) {
				for(int d=0; d<gt.length; d++) {
					w[d] += alphas[i] * lg.get(i)[d];
				}
			}
			t++;
			//System.out.println("w= " + Arrays.toString(w));
			//System.out.println("ap= " + test(ts));
			
			or = cuttingPlane(ts, w);
			gt = (double[]) or[0];
			ct = (Double) or[1];
			
			lg.add(gt);
			lc.add(ct);
		}
		System.out.println("*");
	}
	
	public Object[] cuttingPlane(STrainingSample<List<double[]>, List<Integer>> ts, double[] w) {
		// compute g(t) and c(t)
		double[] gt = new double[w.length];
		double ct = 0;

		List<Integer> y = lossAugmentedInference(ts,w);
		ct += st.delta(st.getLabelsFromRanking(st.gtRanking, st.nbPlus), y);
		double[] at = st.psi(ts.input, y);
		
		for(int d=0; d<w.length; d++) {
			gt[d] += -at[d];
		}
		
		Object[] res = new Object[2];
		res[0] = gt;
		res[1] = ct;
		return res;
	}
	
	public double test(List<STrainingSample<List<double[]>,List<Integer>>> l) {
		return test(l.get(0));
	}
	
	public double test(STrainingSample<List<double[]>,List<Integer>> ts) {
		List<Integer> y = prediction(ts);
		return st.ap(st.getLabelsFromRanking(st.gtRanking, st.nbPlus), y);
	}
	
	public List<Integer> lossAugmentedInference(STrainingSample<List<double[]>,List<Integer>> ts, double[] w){
		// y estimate = arg max_y { Delta(y,yi) + <w,psi(xi,y)>} 
		
		//System.out.println("******************* Loss-augmented inference for ranking ");
		
		List<Integer> sortedPlus = new ArrayList<Integer>();
		List<Integer> sortedMinus = new ArrayList<Integer>();
		
		//List<Double> sortedPlusVal = new ArrayList<Double>();
		//List<Double> sortedMinusVal = new ArrayList<Double>();

		int nbPlus = ((RankingType)st).nbPlus;
		int nbMinus = ts.input.size()-nbPlus;
		
		// Sorting + in descending order of <w ; xi>
		List<Pair<Integer,Double>> pairsPlus = new ArrayList<Pair<Integer,Double>>();
		
		for(int i=0;i< ts.input.size() ; i++){
			if(((RankingType)st).gtLabels.get(i)==1){
				pairsPlus.add(new Pair<Integer,Double>( i , VectorOperations.dot(w, ts.input.get(i))));
			}
		}
		Collections.sort(pairsPlus,Collections.reverseOrder());
		for(int i=0;i< pairsPlus.size() ; i++){
			sortedPlus.add(pairsPlus.get(i).getKey());
			//sortedPlusVal.add(pairsPlus.get(i).getValue());
		}
		
		//System.out.println(" exemples +sorted ");
		
		
		// Sorting - in descending order of <w ; xi>
		List<Pair<Integer,Double>> pairsMinus  = new ArrayList<Pair<Integer,Double>>();
		for(int i=0;i< ts.input.size() ; i++){
			if(((RankingType)st).gtLabels.get(i)==-1){
				pairsMinus.add(new Pair<Integer,Double>( i ,VectorOperations.dot(w, ts.input.get(i))));
			}
		}
		Collections.sort(pairsMinus,Collections.reverseOrder());
		for(int i=0;i< pairsMinus.size() ; i++){
			sortedMinus.add(pairsMinus.get(i).getKey());
			//sortedMinusVal.add(pairsMinus.get(i).getValue());
		}
		
		/*System.out.println("sortedPlus "+sortedPlus);
		System.out.println("sortedMinus "+sortedMinus);
		try {
			System.in.read();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		//System.out.println(" sortedPlus "+sortedPlus+" val="+sortedPlusVal);
		//System.out.println(" sortedMinus "+sortedMinus+" val="+sortedMinusVal);
		
		//System.out.println(" exemples - sorted ");
		
		List<Integer> imaxs = new ArrayList<Integer>(Collections.nCopies(nbMinus, 0));
		// Inserting - examples in the + list
		for(int j=0;j<nbMinus;j++){
			
			//double sum = 0.0;
			
			List<Double> deltasij= new ArrayList<Double>();
			
			for(int k=0;k<nbPlus;k++){
				double skp = pairsPlus.get(k).getValue();
				double sjn = pairsMinus.get(j).getValue();
				double deltaij = val_optj( j, k, skp, sjn,(double)nbPlus,(double)nbMinus);
				deltasij.add(deltaij);
				//sum+= deltaij;
//				//System.out.println("j="+j+" sum "+sum+" skp="+skp+" sjn="+sjn);
//				if(sum>valmax){
//					valmax = sum;
//					imax = k;
//				}
			}
			//System.out.println(" deltasij="+deltasij);
			int imax = 0;
			double valmax = -Double.MAX_VALUE;
			for(int k=0;k<nbPlus;k++){
				double val = 0.0;
				for(int h=k;h<nbPlus;h++){
					val += deltasij.get(h);
				}
				if(val>valmax){
					valmax = val;
					imax = k;
				}
			}
			
			//System.out.println("j="+j+" imax="+imax);
			// Inserting jst - example between (imax)th and (imax+1)st positive 
			
			imaxs.set(j, imax);
			//sortedPlus.add(j,sortedMinus.get(j));
			
		}
		//System.out.println(" FIN LAI - imaxs="+imaxs);
		
		List<Integer> res = Conversion.fusionList(sortedPlus, sortedMinus, imaxs);
		//System.out.println(" FIN LAI - most violated contrsaint : "+res);

		
		return res; 
	}
	private double val_optj(int j, int k, double skp, double sjn, double nbPlus, double nbMinus){
		
		double jj = j+1;
		double kk = k+1;
		
		double val = 1/nbPlus * ( jj / (jj+kk) - (jj-1)/(jj+kk-1))  - 2.0*(skp-sjn)/(nbPlus*nbMinus) ;
		
		return val;
	}
	
	public List<Integer> prediction(STrainingSample<List<double[]>,List<Integer>> ts) {
		// y estimate = arg max_y {<w,psi(xi,y)>} 
		//System.out.println(" Prediction : ");
		List<Integer> res = new ArrayList<Integer>();
		List<Pair<Integer,Double>> pairs = new ArrayList<Pair<Integer,Double>>();
		
		//System.out.println(" Prediction : norm w="+VectorOperations.norm(w));
		
		for(int i=0;i< ts.input.size() ; i++){
//			if(i<10)
//				System.out.println(VectorOperations.dot(w, ts.input.get(i)));
			pairs.add(new Pair<Integer,Double>( i , VectorOperations.dot(w, ts.input.get(i))));
		}
		Collections.sort(pairs,Collections.reverseOrder());
		for(int i=0;i< ts.input.size() ; i++){
//			if(i<10)
//				System.out.println("val="+pairs.get(i).getValue()+" index="+pairs.get(i).getKey());
			res.add(pairs.get(i).getKey());
		}
		
		//System.out.println(" Fin prediction : "+res);
		//return RankingType.convertRankOrdering(res);
		return res;
	}
	
	protected double[] optimMosek(double[][] gram, List<double[]> lg, List<Double> lc, double c) {
		
		mosek.Env env = null;
		mosek.Task task = null;
		
		double[] alphas = null;
		
		try {
			env = new mosek.Env();
			task = new mosek.Task(env,0,0);
			
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
				
				//
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
	
	protected double dot(double[] a, Double[] b) {
		double s = 0;
		for(int i=0; i<a.length; i++) {
			s += a[i] * b[i];
		}
		return s;
	}
	
	protected double matrixProduct(double[] alphas, double[][] gram) {
		// alpha^T*Gramm*alpha
		double[] tmp = new double[alphas.length];
		// tmp = gram * alpha
		for(int i=0; i<gram.length; i++) {
			tmp[i] = VectorOperations.dot(gram[i],alphas);
		}
		double s = VectorOperations.dot(alphas,tmp);
		return s;
	}
	
	public double getLambda() {
		return lambda;
	}
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}
	public double[] getW() {
		return w;
	}
	public void setW(double[] w) {
		this.w = w;
	}
	public int getOptim() {
		return optim;
	}
	public void setOptim(int optim) {
		this.optim = optim;
	}
	public int getCpmax() {
		return cpmax;
	}
	public void setCpmax(int cpmax) {
		this.cpmax = cpmax;
	}
	public int getCpmin() {
		return cpmin;
	}
	public void setCpmin(int cpmin) {
		this.cpmin = cpmin;
	}
	public double getEpsilon() {
		return epsilon;
	}
	public void setEpsilon(double epsilon) {
		this.epsilon = epsilon;
	}
	
	public String toString() {
		return "ssvm_rankAP2_optim_" + optim + "_lambda_" + lambda + "_epsilon_" + epsilon + "_cpmax_" + cpmax + "_cpmin_" + cpmin;
	}
	
	public void save(File file) {
		
		System.out.println("save classifier: " + file.getAbsoluteFile());
		file.getParentFile().mkdirs();
		
		try {
			OutputStream ops = new FileOutputStream(file); 
			OutputStreamWriter opsr = new OutputStreamWriter(ops);
			BufferedWriter bw = new BufferedWriter(opsr);
		
			bw.write("w\n");
			for(int i=0; i<w.length; i++) {
				bw.write(w[i] + "\t");
			}
			bw.write("\nlambda\n" + lambda);
			bw.write("\noptim\n" + optim);
			bw.write("\nepsilon\n" + epsilon);
			bw.write("\ncpmax\n" + cpmax);
			bw.write("\ncpmin\n" + cpmin);
			
			bw.close();
		}
		catch (IOException e) {
			System.out.println("Error parsing file "+ file);
			return;
		}
	}
	
	public void load(File file) {
		
		System.out.println("load classifier: " + file.getAbsoluteFile());
		try {
			InputStream ips = new FileInputStream(file); 
			InputStreamReader ipsr = new InputStreamReader(ips);
			BufferedReader br = new BufferedReader(ipsr);
			
			String ligne;
			ligne=br.readLine(); //"w"
			
			List<List<Double>> list = new ArrayList<List<Double>>();
			int n=0;
			while((ligne=br.readLine()) != null && ligne.compareToIgnoreCase("lambda") != 0) {
				StringTokenizer st = new StringTokenizer(ligne);
				list.add(new ArrayList<Double>());
				while(st.hasMoreTokens()) {
					list.get(n).add(Double.parseDouble(st.nextToken()));
				}
				n++;
			}
			w = new double[list.size()];
			for(int i=0; i<list.size(); i++) {
				for(int j=0; j<list.get(i).size(); j++) {
					w[j] = list.get(i).get(j);
				}
			}
			System.out.println("w " + w.length);
			
			listClass = new ArrayList<Integer>();
			for(int i=0; i<w.length; i++) {
				listClass.add(i);
			}
			
			//ligne=br.readLine(); //"lambda"
			ligne=br.readLine();
			lambda = Double.parseDouble(ligne);
			
			ligne=br.readLine(); //"optim"
			ligne=br.readLine();
			optim = Integer.parseInt(ligne); 
			
			ligne=br.readLine(); //"epsilon"
			ligne=br.readLine();
			epsilon = Double.parseDouble(ligne);
			
			ligne=br.readLine(); //"cpmax"
			ligne=br.readLine();
			cpmax = Integer.parseInt(ligne); 
			
			ligne=br.readLine(); //"cpmin"
			ligne=br.readLine();
			cpmin = Integer.parseInt(ligne); 
			
			br.close();
		}
		catch (IOException e) {
			System.out.println(e);
			System.out.println("Error parsing file " + file);
		}
		
		showParameters();
	}
	
	public void showParameters(){
		System.out.println("----------------------------------------------------------------------------------------");
		System.out.println("Train SSVM Multiclass \tlambda: " + lambda + "\tdim: " + w.length );
		System.out.println("epsilon= " + epsilon + "\t\tcpmax= " + cpmax + "\tcpmin= " + cpmin);
		if(optim == 1) {
			System.out.println("optim " + optim + " - Cutting-Plane 1 Slack - Mosek");
		}
		System.out.println("----------------------------------------------------------------------------------------");
	}

	public RankingType getSt() {
		return st;
	}
	public void setSt(RankingType st) {
		this.st = st;
	}

	/* (non-Javadoc)
	 * @see fr.durandt.jstruct.struct.StructuralClassifier#prediction(java.lang.Object)
	 */
	@Override
	public List<Integer> prediction(List<double[]> x) {
		// TODO Auto-generated method stub
		return null;
	}
}
