# Hybrid SRS Sampling Strategy

Implement a new Simple Random Sampling (SRS) strategy that combines regional stratification with size-proportional allocation. The algorithm will guarantee a minimum of one sample per region to ensure baseline representation. Additional samples will be allocated proportionally to the base-2 logarithm of the region's size. This hybrid approach balances the need for broad geographical coverage with adequate representation of larger areas, preventing massive regions from dominating the dataset while ensuring smaller regions are not ignored.
