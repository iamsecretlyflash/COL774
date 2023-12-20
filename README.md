# COL774 - Machine Learning

This repository contains assignments, lecture notes, exams and tips for the course [COL774 (Machine Learning)](https://www.cse.iitd.ac.in/~parags/teaching/col774/) by Prof. Parag Singla during the Fall Semester 2023-24

COL774 is hands down one of the most challenging and exciting courses at IIT Delhi. Prof. Parag gives significant weightage to exams, which are often extremely difficult but fun to solve if your basic concepts are clear and you have a good imagination ;). Now, there is a pretty simple way to score a 10 in this course. Here are a few tips that will help you throughout this course:

 - Do not listen to people who say machine learning is a hoax or scam. They are just kidding themselves. Machine Learning is a vital skill today. It may be as important as knowing how to access the internet.
 - Keep in touch with the classes. Try to attend as many classes as possible because Prof. Parag is one of the best professors at the institute. His way of teaching is simple but elegant. He explains all concepts quite nicely and also gives points on how to attempt exams.
 - Andrew Ng's Notes are very important. The math in them is nicely explained and very simple to understand.
 - Vector Calculus and Optimisation. You will not survive this course if you do not have a good grasp of vector calculus. As for optimisation, it comes in handy while studying SVMs and PCA. Be thorough with the theory of lagrangians and the KKT Property for Dual problems.
 - PYQs: You can either axe your grade or do PYQs. It's as simple as that.

## Topic wise tips
1) Support Vector Machines
   - Be thorough with Duality Theory.
   - Refer [here](https://stats.stackexchange.com/questions/451868/calculating-the-value-of-b-in-an-svm) for notes on calculating the intercept. (To be updated)
   - What would happen if you use hard-margin SVM with linearly non-seperable data? [Answer](https://www.analyticsvidhya.com/blog/2021/04/insight-into-svm-support-vector-machine-along-with-code/#:~:text=It%20works%20well%20only%20if,SVM%20comes%20to%20the%20rescue) (To be updated)

2) Expectation Maximisation, K-Meeans, GMM and PCA: The unsupervised part is quite easy. I first learnt EM, then GMM and then K-Means because GMM uses EM for learning, and K-Means is essentially a special case of GMM (all covariance matrices are the same; try proving it!). EM is a pretty useful algorithm, and it sort of gives a Bayesian inference vibe. For PCA, all you need is duality theory. Also, do give the class notes a read for PCA. The CS229 notes on PCA do not contain as much information as the class notes.

3) Writing code in PyTorch
   - DO NOT USE CHATGPT!!! I have seen many people getting into trouble because of using ChatGPT. The code might not compile or might not be structured well for your task. Instead, you can use GitHub Co-Pilot for the redundant stuff and go through every line of auto-generated code.
   - PyTorch documentation is your best friend. Every function and feature of PyTorch is exceptionally well documented. There are tons of tutorials available as well. For instance, I found this [tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) on sequence-to-sequence learning using LSTMs extremely helpful.
   - The PyTorch community is extraordinarily vast, and there's a very high chance that somebody has already worked on the code you are working on or something similar.

4) HPC (only for IITD students)
   - Follow everything on this [webpage](https://github.com/kanha95/HPC-IIT-Delhi). The professor will organise a session on HPC, but the tutorial I have linked will be far more helpful than the session.
   - For internet access, do not forget to run "export ftp_proxy = 10.10.78.22:3128" (btech) in the terminal (along with other commands). This way, you can install packages using pip on your login node.
   - The icelake server offers A100 GPU, and it's the best GPU you'll get for the course. Before you ask for GPU access, use "screen" so you do not accidentally lose GPU access. Also, icelake has CUDA 12 (as of 6th December 2023), so you will need to install the latest PyTorch using pip to use GPU while training. "module load" gives an older version of PyTorch which is not built with the latest CUDA.
   - (If you can't tell already, CUDA is a headache)

## What to do once you are done with COL774?

1) You can take COL775(DL) and/or NLP(ELL881/COL772) in the following sem. Very interesting and the courses will cover most of the currently used techniques in DL, ML and AI.
2) Read Pattern Recognition and Machine Learning (Bishop). Absolutely amazing book. Big time Brainfuck but absolutely amazing.
