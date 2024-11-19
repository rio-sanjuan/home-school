Rather than focusing on predicting a single point in time of an event, the prediction step in survival analysis often focuses on predicting a function: either the *survival* or *hazard* function.
### Survival Function

The survival function $S(t)$ returns the probability of survival beyond time $t$, i.e., $S(t) = P(T > t)$.
### Hazard Function

The hazard function $h(t)$ denotes an approximate probability (it is not bounded from above) that an event occurs in the small time interval $\left[t, t + 𝛥 t\right]$, under the condition that an individual would remain event-free up to time $t$: $$ h(t) = \lim_{𝛥 t \to 0} \frac{P(t\leq T < t + 𝛥 t \lvert T \geq t)}{𝛥 t} \geq 0.$$Alternative names for the hazard function are *conditional failure rate*, *conditional mortality rate*, or *instantaneous failure rate*. In contrast to the survival function, which describes the absence of an event, the hazard function provides information about the occurrence of an event.

### Cumulative Hazard Function

The cumulative hazard function $H(t)$ is the integral over the interval $[0, t]$ of the hazard function: $$ H(t) = \int_0^t h(u)\;du.$$
