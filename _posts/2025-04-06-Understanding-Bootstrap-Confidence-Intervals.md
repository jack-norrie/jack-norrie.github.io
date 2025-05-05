---
header:
  image: /assets/images/2025-04-06-Understanding-Bootstrap-Confidence-Intervals.png
---

## Introduction

Recently I set out to understand Bias-Corrected Accelerated (BCa) bootstrap confidence intervals. I was aware that it was the "gold standard" for bootstrapping and that many statistical packages used it as their default. However, I never understood the need for it relative to simpler approaches like the percentile method, which seemed more intuitive

However, this intuitive understanding was misplaced, it turns out the BCa method can be framed as a natural progression from the percentile method. This progression can actually be shown to improve the accuracy of the confidence interval from first order to second order, meaning that the confidence interval coverage converges to the nominal rate quicker. Specifically, the convergence of a reported $1-\alpha$ level confidence intervals goes from

$$\mathbb{P}(\theta \in C_{\text{percentile}}) = 1-\alpha + O\left(\frac{1}{\sqrt{n}}\right)$$

to a rate of

$$\mathbb{P}(\theta \in C_{BC_a}) = 1-\alpha + O\left(\frac{1}{n}\right)$$

which makes a big difference for smaller sample size problems.

## Bootstrap

First, lets have a quick refresher on what bootstrapping is. Suppose you have some fixed size data sample $x=(x\_1, x\_2, \dots, x\_n)$ from an unknown data generation process $X\_1, X\_2, \dots, X\_n \overset{i.i.d.}{\sim} F$, and you use this sample to generate a statistic $\hat{\theta}\_n = t(X\_1, X\_2, \dots, X\_n)=t(x)$. Now suppose we want to investigate the statistical properties of $\hat{\theta}\_n$. For example, we might be interested in $\mathbb{V}_F(\hat{\theta}\_n)$, or more generally $\mathbb{E}(g(\hat{\theta}\_n))$. However, our statistic's distribution $G\_F$ is a function of our unknown distribution $F$. therefore without the unknown distribution we are at a dead end.

This is where the ingenuity of the bootstrap method comes into play. What if instead of using $F$ we used $\hat{F}\_n$? In other words, what if instead of trying to calculate these expected values relative to the unknown distribution, we instead use our empirical distribution associated with adding $\frac{1}{n}$ point masses at each of the data points in the sample. The motivation for this being that, by the Glivenko–Cantelli theorem, $\hat{F}\_n \rightarrow F$.

The power of this approximation is that it means we can estimate the expected values via simulation. The key insight, is that sampling from $\hat{F}\_n$ is the same as resampling our original sample with replacement to generate a new size $n$  sample. We can then use this resampled data $x^\* =(x\_{1, b}^*, x\_{2, b}^\*, \dots, x\_{n, b}^\*)$ to generate a bootstrap replicate of our statistic $\hat{\theta}\_n^\*=t(x^\*)$. If we generate $B$ such bootstrap replicates then, by the law of large numbers, the sample mean of these will converge to the expected value of our statistic under $\hat{F}\_n$.

$$\frac{1}{B}\sum^B_{b=1}g(\theta^*_{n, b})\rightarrow\mathbb{E}_{\hat{F}_n}(g(\hat{\theta}))$$

Importantly this means that we can estimate moments such as:

$$\bar{\theta}_B = \frac{1}{B}\sum^B_{b=1}\theta^*_{n, b}\rightarrow\mathbb{E}_{\hat{F}_n}(\hat{\theta})$$
$$\frac{1}{B}\sum^B_{b=1}(\theta^*_{n, b})^2\rightarrow\mathbb{E}_{\hat{F}_n}(\hat{\theta}^2)$$

Which in turn means that we can estimate the variance as:

$$v_{\text{boot}} = \frac{1}{B}\sum_{b=1}^B(\hat{\theta}_n^* - \bar{\theta}_B) = \frac{1}{B}\sum^B_{b=1}(\theta^*_{n, b})^2 - \left(\frac{1}{B}\sum^B_{b=1}\theta^*_{n, b}\right)^2$$
$$v_{\text{boot}}\overset{p}{\rightarrow} \mathbb{E}_{\hat{F}_n}(\hat{\theta}^2) - (\mathbb{E}_{\hat{F}_n}(\hat{\theta}))^2 = \mathbb{E}_{\hat{F}_n}(\hat{\theta}_n)$$

Furthermore, we can estimate the quantile function using

$$\hat{G}(t) = \frac{1}{B}\sum^B_{b=1} \mathbb{I}(\theta^*_{n, b} \leq t) \overset{p}{\rightarrow}\mathbb{E}_{\hat{F}_n}(\mathbb{I}(\hat{\theta}_n \leq t))=\mathbb{P}_{\hat{F}_n}(\hat{\theta}_n \leq t) = G_{\hat{F}_n}(t)$$

To summarise, bootstrapping allows us to estimate expected values of functions of our statistic via simulation of the empirical distribution. This can be seen as a two stage approximation procedure:

$$\mathbb{E}_F(g(\theta)) \overset{n\rightarrow \infty}\approx \mathbb{E}_{\hat{F}_n}(g(\theta))\overset{B \rightarrow \infty}{\approx} \frac{1}{B}\sum_{b=1}^Bg(\theta^*_{n, b}) $$

## Confidence Intervals

Naturally we might now ask whether we can leverage our toolkit, namely being able to estimate bootstrap quantiles and variances, to calculate confidence intervals for estimators.

if we can generate a confidence interval of the form:

$$\hat{\theta}_n \in C_{\hat{F}_n^*}(\hat{\theta}_n) \quad s.t. \quad \mathbb{P}\left(\hat{\theta}_n^* \in C_{\hat{F}_n}(\hat{\theta}_n))\right)=1-\alpha$$

Then it follows that in the limit of $n\rightarrow \infty$, by virtue of $\hat{F}\_n \rightarrow F$ and $\hat{\theta}\_n \rightarrow \theta$, we would naturally get an approximation for a confidence interval of the form:

$$\theta \in C_F(\hat{\theta}_n) \quad s.t. \quad \mathbb{P}\left(\theta \in C_F(\hat{\theta}_n)\right)=1-\alpha$$

Naturally this leads to the observation that we can general make the correspondence  

$$\theta \leftrightarrow \hat{\theta}_n$$
$$\hat{\theta}_n\leftrightarrow \hat{\theta}_n^*$$

when moving between $F$ and $\hat{F}\_n$. This is sometimes referred to as the "bootstrap principle".

However, for finite $n$, under certain assumptions different confidence interval generation processes will converge to this asymptotic correspondence faster than others, and therefore reach their nominal level of coverage faster. The methods below have been grouped as either making assumptions about pivotal quantities or normality.

## Pivotal Assumptions

Each of the interval construction methods below will work best when their proposed transformation $g\_{\theta}$ on  $\hat{\theta}$ gives rise to a distribution that does not depend on any unknown parameters. In other words we have transformations of the form:

$$g_{\theta}(\hat{\theta}) \sim G \quad \forall\theta \implies g_{\hat{\theta}}(\hat{\theta}^*) \sim G$$

Rather than:

$$g_{\theta}(\hat{\theta}) \sim G_{\theta} \quad \forall\theta$$

Importantly, under these assumptions, we can use our bootstrap distribution to calculate approximations for quantiles of our non-bootstrap distribution. Obviously this is highly unlikely to be exactly true, but for certain problems some of these bootstrap confidence interval estimation procedure's assumptions will be approximately true to a greater degree than others.

### Basic Bootstrap

For basic bootstrap we assume that the difference between the quantity $t(\hat{F}\_n) - t(F)$ is pivotal, specifically this means that

$$R_n = \hat{\theta}_n - \theta \sim H\quad \ \forall \theta \implies R_n^* = \hat{\theta}_n^* - \hat{\theta}_n \sim H$$

Which means that

$$\mathbb{P}\left(H^{-1}\left(\frac{\alpha}{2}\right) \leq \hat{\theta}_n - \theta \leq  H^{-1}\left(1-\frac{\alpha}{2}\right)\right)=1-\alpha$$
$$\mathbb{P}\left(\hat{\theta}_n - H^{-1}\left(1-\frac{\alpha}{2}\right) \leq \theta \leq  \hat{\theta}_n - H^{-1}\left(\frac{\alpha}{2}\right)\right)=1-\alpha$$

i.e. the following is a $1-\alpha$ confidence interval

$$\left(\hat{\theta}_n - H^{-1}\left(1-\frac{\alpha}{2}\right), \hat{\theta}_n - H^{-1}\left(\frac{\alpha}{2}\right)\right)$$

However, as has been stressed, we cannot directly calculate the distribution $G$ of our statistic $\hat{\theta}\_n$. However, under our assumptions that the bootstrap pivots follow the same distribution $G$, it follows that we can use the bootstrap quantile function to estimate the above quantities. Specifically, we can calculate bootstrap replicates of our pivotal quantiles of the form

$$R^*_{n, b}=\theta^*_{n, b}-\hat{\theta}_n$$

And then get the $\frac{\alpha}{2}$ and $1 - \frac{\alpha}{2}$ quantiles. Furthermore,  we can simplify the form of this confidence interval purely in terms of our bootstrap resamples quantiles. Notice that $R^\*\_{n, b}$ is monotonically increasing in $\theta^\*\_{n, b}$, which means that the estimated quantiles of the pivot $\hat{H}^{-1}(\beta)$ are related to estimated quantiles of the statistic $\hat{G}^{-1}(\beta)$ via

$$\hat{H}^{-1}(\beta) = \hat{G}^{-1}(\beta)-\hat{\theta}_n$$

Which means we can restate our bootstrap confidence interval as

$$C_\text{pivot}=\left(2\hat{\theta}_n - \hat{G}^{-1}\left(1-\frac{\alpha}{2}\right), 2\hat{\theta}_n - \hat{G}^{-1}\left(\frac{\alpha}{2}\right)\right)$$

### Bootstrap-t Intervals

The key assumption of the pivot bootstrap, in terms of rapid convergence, is that the quantity $t(\hat{F}\_n) - t(F)$ is pivotal. However, we know from classical statistics problems that this is rarely the case, it is far more common to deal with pivotal quantities of the form

$$\frac{t(\hat{F}_n) - t(F)}{\sqrt{\mathbb{V}(t(\hat{F}_n))}}$$

This leads to a pivotal quantity that is akin to the t-statistic. However, unlike a classical t-test, which usually makes strong distributional assumptions, the bootstrap-t method only demands that this quantity is pivotal. The key assumption therefore is that

$$t_n = \frac{\hat{\theta}_n - \theta}{\sigma} \sim H\quad \ \forall \theta \implies t_n^* = \frac{\hat{\theta}_n^* - \hat{\theta}_n}{\sqrt{v_{\text{boot}}}} \sim H$$

Which means that

$$\mathbb{P}(t^*_{\frac{\alpha}{2}} \leq t_n \leq t^*_{1-\frac{\alpha}{2}})=1-\alpha$$
Therefore a $1-\alpha$ confidence interval for $t_n$ is
$$C_{\text{bootstrap-t}}=\left(\hat{\theta}_n-\hat{H}^{-1}\left(1-\frac{\alpha}{2}\right)\sqrt{v_{\text{boot}}}, \hat{\theta}_n-\hat{H}^{-1}\left({\frac{\alpha}{2}}\right)\sqrt{v_{\text{boot}}}\right)$$

>[!WARNING]
>It should be stressed, we are not reading these quantile values from a t-table, they are estimated from the bootstrap distribution using the bootstrap ECDF $\hat{H}$  for the pivotal quantity, i.e. bootstrap-t statistic.

## Normality Assumptions

The below methods all make normality assumptions in some way, whether this is that some statistic is normally distributed or that there exists some procedure that we can perform to make the statistic normally distributed. The methods have been grouped together, because they can be seen as building on each other, at each stage loosening the assumptions and therefore having optimal convergence for a wider set of problems.

### Standard  Intervals

The simplest method to generate a bootstrap confidence interval is to use

$$C_{\text{standard}} = (\hat{\theta}_n - z_{\alpha/2}\sqrt{v_{\text{boot}}}, \hat{\theta}_n + z_{\alpha/2}\sqrt{v_{\text{boot}}})$$

However,  this makes very strong assumptions. Specifically that the quantity below is pivotal and follows a normal distribution:

$$\frac{\hat{\theta}_n - \theta}{\sigma} \sim N(0, 1)\quad \ \forall \theta \implies \frac{\hat{\theta}_n^* - \hat{\theta}_n}{\sqrt{v_{\text{boot}}}} \sim N(0, 1)$$

Which is obviously a much stronger assumption than the bootstrap-t method, and will not work in a variety of circumstances, especially for small $n$

### Percentile Intervals

The most intuitive way of deriving a confidence interval is

$$C_{\text{percentile}}=(\theta^*_{\frac{\alpha}{2}}, \theta^*_{1-\frac{\alpha}{2}})$$

And indeed, this will asymptotically converge to a confidence interval at the nominal level as the bootstrap distribution converges to the true statistic sampling distribution. However, further analysis can be performed to give further justifications for such an interval, and conditions for which we expect minimal approximation error.

Suppose there exists a monotone transformation $m$, which can be used on our raw statistic as follows:

$$\phi=m(\theta)$$
$$\hat{\phi}_n=m(\hat{\theta}_n)$$
$$\hat{\phi}_n^*=m(\hat{\theta}_n^*)$$
Such that
$$\hat{\phi}_n\sim N(\phi, \sigma^2)\quad \forall \phi \implies \hat{\phi}_n^*\sim N(\hat{\phi}_n, \sigma^2)$$

>[!TIP]
>We do not claim to know $m$, we only require that such a transformation exists.

Which means we can form pivotal quantities

$$\frac{\hat{\phi}_n-\phi}{\sigma}\sim N(0, 1)\quad \forall \implies \frac{\hat{\phi}_n^*-\hat{\phi}_n}{\sigma}\sim N(0, 1)$$

Therefore, since

$$\mathbb{P}\left(z_{\frac{\alpha}{2}} \leq \frac{\hat{\phi}_n-\phi}{\sigma} \leq z_{1-\frac{\alpha}{2}}\right)=1-\alpha$$

It follows that

$$(\hat{\phi}_n - z_{1-\frac{\alpha}{2}} \sigma, \hat{\phi}_n - z_{\frac{\alpha}{2}} \sigma)$$
is a $1-\alpha$ confidence interval for $\phi$.

Importantly, due to the symmetry of the normal distribution, we can also write this:

$$(\hat{\phi}_n + z_{\frac{\alpha}{2}} \sigma, \hat{\phi}_n + z_{1-\frac{\alpha}{2}} \sigma)$$

>[!TIP]
>This is the critical step as to why the percentile method works, i.e. the symmetry of the normal distribution. In fact we could restart our analysis with:
>
>$$\frac{\hat{\phi}_n-\phi}{\sigma}\sim H\quad \forall \implies \frac{\hat{\phi}_n^*-\hat{\phi}_n}{\sigma}\sim H$$
>
>Where $H$ is symmetric. Therefore the percentile method actually has optimal convergence properties for a wider range of scenarios than this analysis would imply.

We now make the observation that the bootstrap distribution's quantiles for the transformed distribution are equal to

$$H(\beta)=\hat{\phi}_n + z_{\beta} \sigma$$

which means that the confidence interval above can be restated

$$\left(H^{-1}\left(\frac{\alpha}{2}\right), H^{-1}\left(1-\frac{\alpha}{2}\right)\right)$$

Furthermore, since we are dealing with a monotone transform it follows that

$$G(t)=\mathbb{P}(\hat{\theta}_n^* \leq t) = \mathbb{P}(m(\hat{\theta}_n^*) \leq m(t))=\mathbb{P}(\hat{\phi}_n^* \leq m(t))=H(m(t)),$$

Which means that

$$G(t) = \beta \implies H(m(t)) = \beta \implies t = m^{-1}(H^{-1}(\beta))$$
$$\implies G^{-1}=m^{-1} \circ H^{-1}$$

Therefore when we map our confidence interval back to $\theta$ space via $m^{-1}$, the result on applying this transform to quantiles in $\phi$ is to get back the corresponding quantiles in $\theta$, i.e. our confidence interval in $\theta$ space is

$$\left(G^{-1}\left(\frac{\alpha}{2}\right), G^{-1}\left(1-\frac{\alpha}{2}\right)\right)$$

Finally, we can use our trick of estimating quantiles using our bootstrap distribution to get an estimated confidence interval

$$\left(\hat{G}^{-1}\left(\frac{\alpha}{2}\right), \hat{G}^{-1}\left(1-\frac{\alpha}{2}\right)\right),$$

thus giving justification for the percentile method and conditions under which it should converge optimally.

### Bias Corrected (BC) Intervals

Our previous approach worked on the assumption that there was a monotone normalising transformation. However, it turns out that if our statistic is median biased, then we know for certain that no such transformation exists. This is relatively simple to show. If we truly have a transformation producing $m(\hat{\phi}\_n) \sim N(\phi, \sigma^2)\quad \forall \phi$ then it follows that:

$$\mathbb{P}(\hat{\phi}_n \leq \phi)=0.5$$

Monotonicity would then also imply that

$$\mathbb{P}(\hat{\theta}_n \leq \theta)=G(\theta)=0.5$$

Fortunately, we can use simulation on our bootstrap distribution to estimate the median bias. Specifically, we can use the bootstrap estimate the CDF of our statistic

$$p_0 = \hat{G}(\hat{\theta}) = \frac{\#(\hat{\theta}^* \leq \hat{\theta})}{B}, $$

and check whether it is equal to 0.5.

If we find that our estimator is median biased then we can be certain that there does not exist a monotone normalising transformation around mean zero. However, we could now relax our condition and allow there to be a monotone normalising transformation around some offset $-z\_0\sigma$.  

$$\hat{\phi}_n-\phi \sim N(-z_0 \sigma, \sigma^2) \quad \forall \phi \implies\hat{\phi}_n^*-\hat{\phi}_n \sim N(-z_0 \sigma, \sigma^2)$$
$$\frac{\hat{\phi}_n-\phi + z_0\sigma}{\sigma} \sim N(0, 1) \quad \forall \phi \implies \frac{\hat{\phi}_n^*-\hat{\phi}_n+z_0\sigma}{\sigma} \sim N(0, 1)$$

We can now construct a confidence interval (again using the symmetry of the normal distribution):

$$=\mathbb{P}(z_{\frac{\alpha}{2}} \leq \frac{\hat{\phi}_n -\phi +z_0\sigma}{\sigma} \leq z_{1-\frac{\alpha}{2}} ) = 1 - \alpha$$
$$=\mathbb{P}(-z_{1-\frac{\alpha}{2}} \leq \frac{ \phi-\hat{\phi}_n -z_0\sigma}{\sigma} \leq -z_{\frac{\alpha}{2}} ) = 1 - \alpha$$
$$=\mathbb{P}(z_{\frac{\alpha}{2}} \leq \frac{ \phi-\hat{\phi}_n -z_0\sigma}{\sigma} \leq z_{1-\frac{\alpha}{2}} ) = 1 - \alpha$$
$$=\mathbb{P}(\hat{\phi}_n + z_0\sigma +z_{\frac{\alpha}{2}}\sigma \leq \phi\leq \hat{\phi}_n + z_0\sigma +z_{1-\frac{\alpha}{2}}\sigma ) = 1 - \alpha$$

However, unlike before we cannot immediately identify these as the quantiles of $\hat{\phi}^*$, these would have the form

$$H^{-1}(\beta) = \hat{\phi}_n - z_0\sigma +z_{\beta}\sigma $$

Instead we notice that confidence interval above is of the form

$$(\hat{\phi}_n + z_0\sigma +z_{\frac{\alpha}{2}}\sigma, \hat{\phi}_n + z_0\sigma -z_{\frac{\alpha}{2}}\sigma)=\left(H^{-1}\left(\frac{\alpha}{2}\right)+2z_0\sigma, H^{-1}\left(1-\frac{\alpha}{2}\right)+2z_0\sigma\right)$$

Therefore, we expect the bias correction to have  shifting effect on the quantiles

Lets now investigate this more precisely. We want to know the quantile associated with each of the above confidence interval bounds.  

$$H(\hat{\phi}_n + z_0\sigma \pm z_{\frac{\alpha}{2}}\sigma)=\mathbb{P}(\hat{\phi}_n^* \leq \hat{\phi}_n + z_0\sigma \pm z_{\frac{\alpha}{2}}\sigma)$$
$$= \mathbb{P}\left(\frac{\hat{\phi}_n^* - \hat{\phi}_n +z_0\sigma}{\sigma}\leq + 2z_0\pm z_{\frac{\alpha}{2}}\right)=\Phi(2z_0 \pm z_{\frac{\alpha}{2}})$$
$$\hat{\phi}_n + z_0\sigma \pm z_{\frac{\alpha}{2}}\sigma=H^{-1}(\Phi(2z_0 \pm z_{\frac{\alpha}{2}}))$$

Therefore, our confidence interval is of the form

$$(\hat{\phi}_n + z_0\sigma +z_{\frac{\alpha}{2}}\sigma, \hat{\phi}_n + z_0\sigma -z_{\frac{\alpha}{2}}\sigma)=(H^{-1}(\Phi(2z_0 + z_{\frac{\alpha}{2}})), H^{-1}(\Phi(2z_0 - z_{\frac{\alpha}{2}})))$$

Which can be transformed to $\theta$ space via the transform $m^{-1}$ to reveal

$$(G^{-1}(\Phi(2z_0 + z_{\frac{\alpha}{2}})), G^{-1}(\Phi(2z_0 - z_{\frac{\alpha}{2}})))$$

However, currently $z\_0$ is a nuisance parameter and is stopping us from mapping our confidence interval back to $\theta$ space. Lets see if we can estimate it by revisiting the test we performed at the start of this analysis, now with the caveat that we have this added parameter.

$$H(\phi)=\mathbb{P}(\hat{\phi}_n\leq \phi) = \mathbb{P}\left(\frac{\hat{\phi}_n-\phi+z_0\sigma}{\sigma}\leq z_0\right)=\Phi(z_0)$$

Which means that:

$$z_0 =\Phi^{-1}\left(H(\phi)\right)$$

By monotonicity:

$$z_0= \Phi^{-1}\left(G(\theta)\right)$$

Which can be estimated using the bootstrap distribution as

$$\hat{z}_0= \Phi^{-1}\left(\hat{G}(\hat{\theta})\right)=\Phi^{-1}(p_0)$$

Finally using our bootstrap estimated $\hat{G}$ and $z\_0$ we can get estimates for these $1-\alpha$ confidence intervals

$$(\hat{G}^{-1}(\Phi(2\hat{z}_0 + z_{\frac{\alpha}{2}})), \hat{G}^{-1}(\Phi(2\hat{z}_0 - z_{\frac{\alpha}{2}})))$$

>[!TIP]
>Similar to percentile intervals we could have replaced the normal assumption to a symmetric assumption. However, in this case, this would change the final form of our confidence interval transformation to to involve non-normal quantiles. The popularity of the normal assumption reflects the fact that these quantiles are easily calculable via statistical software.

### Bias Corrected Accelerated Confidence Intervals

As we saw from the BC analysis, loosening our ideal transformation allowed us to use our procedure in a wider set of problems while enjoying optimal convergence properties. Indeed the procedure detailed below actually enjoys second order accuracy, which makes it especially powerful for bootstrapping on smaller sample sizes.

Bias-Corrected Accelerated bootstrap (BCa) loosens the requirement for constant variance, allowing the standard deviation to depend linearly on the unknown parameter. Specifically, assume  there exists some monotone transformation that produces

$$\hat{\phi} \sim N(\phi - z_0\sigma_{\phi}, \sigma^2_{\phi}), \quad \sigma_{\phi}=1+a\phi, \quad \forall \phi \implies \hat{\phi}^*_n \sim N(\hat{\phi}_n - z_0\sigma_{\hat{\phi}_n}, \sigma^2_{\hat{\phi}_n})$$

Where BC can  be seen as the special case of $a=0$ and the percentile method can be seen as the special case of $z\_0=0$ and $a=0$.

After an even more elaborate analysis than the BC method, we arrive at an estimated confidence interval:

$$\hat{\theta}_{BCa}[\beta] = \hat{G}^{-1}\left(\Phi\left(\hat{z_0} + \frac{\hat{z_0}+z_{\beta}}{1-\hat{a}(z_0+z_{\beta})}\right)\right)$$

For non-parametric problems $a$ can be estimated using jackknife resampling:

$$\hat{a}=\frac{1}{6}\frac{\sum^n_{i=1}(\hat{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^3}{\left(\sum^n_{i=1}(\hat{\theta}_{(\cdot)} - \hat{\theta}_{(i)})^2\right)^{\frac{3}{2}}}$$

and $z_0$ is estimated as before:

$$\hat{z}_0=\Phi^{-1}(p_0)$$
$$p_0 = \hat{G}(\hat{\theta}) = \frac{\#(\hat{\theta}^* \leq \hat{\theta})}{B}, $$

>[!TIP]
>For best results $|z\_0|$ and $|a|$ should be small

## References

* Wasserman L. All of statistics: a concise course in statistical inference. Springer Science & Business Media; 2013 Dec 11.
* Efron B, Hastie T. Computer age statistical inference, student edition: algorithms, evidence, and data science. Cambridge University Press; 2021 Jun 17.
* Efron B. The jackknife, the bootstrap and other resampling plans. Society for industrial and applied mathematics; 1982 Jan 1.
* Efron B. Better bootstrap confidence intervals. Journal of the American statistical Association. 1987 Mar 1;82(397):171-85.
