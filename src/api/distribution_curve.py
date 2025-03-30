import numpy as np
import pandas as pd
from typing import Tuple
from scipy.stats import skewnorm


class AgeDistribution:

    def __init__(
        self,
        pop_0_14: float,
        pop_15_64: float,
        pop_65_above: float,
    ):
        self.pop_0_14: float = pop_0_14
        self.pop_15_64: float = pop_15_64
        self.pop_65_above: float = pop_65_above

    def get_params_for_age_distribution_curve(
        self,
        max_age: int = 100,
    ) -> Tuple[float, float, float]:
        """
        Gets the parameters of the distribution curve for the age distributop

        Args:
            pop_0_14 (float): Percentage of population aged 0-14 years.
            pop_15_64 (float): Percentage of population aged 15-64 years.
            pop_65_above (float): Percentage of population aged 65-max_age years.
            max_age (int, optional): Gets the estimated max age of the people. Defaults to 100.

        Returns:
            Tuple[float, float, float]: mean age, standard deviation, skewness
        """

        # Convert to fractions
        total_population = self.pop_0_14 + self.pop_15_64 + self.pop_65_above
        p_0_14 = self.pop_0_14 / total_population
        p_15_64 = self.pop_15_64 / total_population
        p_65_above = self.pop_65_above / total_population

        # Midpoints of each group
        mid_0_14 = 7
        mid_15_64 = 40
        mid_65_above = (max_age + 65) / 2

        # Compute mean (μ) by weighted mean
        mean_age = (
            (mid_0_14 * p_0_14) + (mid_15_64 * p_15_64) + (mid_65_above * p_65_above)
        )

        # Compute standard deviation (σ)
        std_dev = np.sqrt(
            (p_0_14 * (mid_0_14 - mean_age) ** 2)
            + (p_15_64 * (mid_15_64 - mean_age) ** 2)
            + (p_65_above * (mid_65_above - mean_age) ** 2)
        )

        # Estimate skewness (α)
        median_age = 40  # Approximate median from the largest group
        skewness = 3 * (mean_age - median_age) / std_dev

        return mean_age, std_dev, skewness

    def get_age_probabilities(
        self,
        max_age: int = 100,
    ) -> pd.DataFrame:
        """
        Based on created skewed normal distribution curve gets the probablity from age 0 to `max_age`.

        Args:
            max_age (int, optional): Gets the estimated max age of the people. Defaults to 100.

        Returns:
            out: pd.DataFrame: Dataframe with calculated distributions.
        """
        # Define skew-normal distribution parameters
        mean_age, std_dev, skewness = self.get_params_for_age_distribution_curve(
            max_age=max_age
        )

        # Generate age range
        ages = np.arange(0, max_age + 1, 1)  # from 0 to max_age (included)

        # Compute skew-normal probability density function (PDF)
        pdf = skewnorm.pdf(ages, skewness, loc=mean_age, scale=std_dev)

        # Normalize to ensure total probability sum = 1
        pdf /= np.sum(pdf)

        # Create dataframe to return
        curve_df = pd.DataFrame(columns=["age", "probability"])

        # Fill the values for the curve
        for age, prob in zip(ages, pdf):
            # age_row = {"age": age, "probability": prob}
            age_row = [age, prob]
            curve_df.loc[len(curve_df)] = age_row

        return curve_df
