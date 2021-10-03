Codes here (under djkp) are mostly from https://github.com/ahmedmalaa/discriminative-jackknife. 

We fixed a few bugs to ensure fair comparison of some baselines:
1. In the original Deep Ensemble paper, the authors advise to use NLL loss to learn uncertainty. 
We fixed that in this implementation (called "DE_Correct")
   
2. In `run_baseline`, the z_score was incorrect. 

3. In computing the influence function, there were some bugs that prevents the codes from running, which have been fixed.