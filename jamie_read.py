#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools to assist with analysing multi source lightcurve data.
Trying to generalise/genericify different data sets.
@author: J. Soon
@email: jamie.soon@anu.edu.au
@original: 14/11/2022
@refactored: 06/02/2023
"""
# STDLIB
import pathlib
from dataclasses import dataclass, field

# 3rd Party
import matplotlib.pyplot as plt
import numpy as np
#import as tomllib to eventually switch to the standard python package with python 3.11
import tomli as tomllib

#Local
config_file = "config.toml"

with open(config_file, "rb") as f:
    config = tomllib.load(f)



# Create dataclasses as it allows people to play with the data as opposed to fixed functions
@dataclass
class SingleLightcurve:
    ra: list = field(default_factory=list)
    dec: list = field(default_factory=list)
    time: list = field(default_factory=list)
    magnitude: list = field(default_factory=list)
    flux: list = field(default_factory=list)
    flux_error: list = field(default_factory=list)
    time_offset: float = 0
    colour: tuple = (0, 0, 0)

    def __post_init__(self):
        if len(set(map(len, [self.ra, self.dec, self.time, self.magnitude]))) != 1:
            raise ValueError("Measurements needs to be of same length")


class Lightcurves:
    """
    A class used to represent a collection of lightcurves

    ...

    Attributes
    ----------
    ra : double
        The RA in degrees
    dec : double
        The Dec in degrees
    name : str, optional
        The name of the object (default is "ra, dec")
    verbose : int, optional
        Verbosity (default is 0)

    Methods
    -------
    add_lightcurve()
        Adds a single lightcurve to the lightcurve collection.
    load_lightcurves_from_db()
        Load lightcurves from a database.
    load_lightcurve_from_csv()
        Loads a single lightcurve from a csv file.
    load_lightcurves_from_folder()
        Loads all lightcurves that are csv files in a folder.
    plot()
        A default plot to visualise.
    save()
        Save lightcurve collection in a single format.
    """

    def __init__(
        self, ra: float, dec: float, name: str = None, verbose: int = 0
    ) -> None:
        self.ra = ra
        self.dec = dec
        if name is None:
            self.name = f"{ra}, {dec}"
        else:
            self.name = name
        self.verbose = verbose
        self.data = {}

        self.survey_data = config["surveys"]

    def __str__(self) -> str:
        return f"Collection of lightcurves at RA:{self.ra}, Dec:{self.dec}, with object name {self.name}"

    def add_lightcurve(
        self,
        survey: str,
        filter_name: str,
        ra: list,
        dec: list,
        time: list,
        magnitude: list,
        time_offset: float,
        colour: tuple,
    ) -> None:
        if survey not in self.data:
            self.data[f"{survey}"] = {}
        self.data[f"{survey}"][f"{filter_name}"] = SingleLightcurve(
            ra=ra,
            dec=dec,
            time=time,
            magnitude=magnitude,
            time_offset=time_offset,
            colour=colour,
        )

    def load_lightcurves_from_db(self):
        raise NotImplementedError

    def load_lightcurve_from_csv(self, input_file: str, survey: str = None) -> None:
        input_file = pathlib.Path(input_file)
        lightcurve_data = np.genfromtxt(
            input_file, delimiter=",", dtype=None, encoding=None, names=True
        )

        if survey is None:
            easy_survey = self.parse_filename(input_file.stem)
            # Convert survey "easy name" in filename to config survey name
            for surveys in self.survey_data:
                if self.survey_data[surveys]["easy_name"] == easy_survey:
                    survey = surveys

        try:
            time_column = self.survey_data[survey]["time_column"]
            time_offset = self.survey_data[survey]["time_offset"]
        except KeyError:
            print(
                f"Incorrect survey name: {survey}, change input survey name or check config file"
            )
            return

        for filters in self.survey_data[survey]["filters"]:
            if "value" in self.survey_data[survey]["filters"][filters]:
                filter_column = self.survey_data[survey]["filters"][filters]["column"]
                filter_value = self.survey_data[survey]["filters"][filters]["value"]

                if type(filter_value) == list:
                    combined_mask = np.full(
                        np.shape(lightcurve_data[filter_column]), False
                    )
                    for value in filter_value:
                        boolean_mask = lightcurve_data[filter_column] == value
                        combined_mask = np.logical_or(combined_mask, boolean_mask)
                else:
                    combined_mask = lightcurve_data[filter_column] == filter_value

                selected_data = lightcurve_data[combined_mask]
            else:
                selected_data = lightcurve_data

            magnitude_column = self.survey_data[survey]["filters"][filters]["magnitude"]
            colour = tuple(self.survey_data[survey]["filters"][filters]["colour"])

            if self.verbose > 9:
                print(survey)
                print(filters)
                print(selected_data)
                print(time_offset)
                print(colour)

            self.add_lightcurve(
                survey,
                filters,
                selected_data["ra"],
                selected_data["dec"],
                selected_data[time_column],
                selected_data[magnitude_column],
                time_offset,
                colour,
            )

        return

    def load_lightcurves_from_folder(
        self, input_folder: str, glob: str = "*.csv"
    ) -> list:
        input_folder = pathlib.Path(input_folder)
        files = sorted(input_folder.glob(glob))

        for indiviudal_file in files:
            self.load_lightcurve_from_csv(indiviudal_file)

        return files

    def parse_filename(
        self, filename: str, delimiter: str = "_", number: int = -1
    ) -> str:
        split_filename = filename.split(delimiter)
        survey = split_filename[number]

        return survey

    def plot(self, ax: plt.Axes) -> None:
        ax.clear()

        ax.set_xlabel("Time (JD)")
        ax.set_ylabel("Magnitude")
        if not ax.yaxis_inverted():
            ax.invert_yaxis()

        ax.ticklabel_format(useOffset=False, style="plain")

        for surveys in self.data:
            for filters in self.data[surveys]:
                label = f"{surveys} {filters}"
                colour = tuple(
                    color / 255 for color in self.data[surveys][filters].colour
                )
                ax.scatter(
                    self.data[surveys][filters].time
                    + self.data[surveys][filters].time_offset,
                    self.data[surveys][filters].magnitude,
                    color=colour,
                    label=label,
                )

        ax.set_title(f"Name: {self.name}, RA:{self.ra:.5f}, Dec:{self.dec:.5f}")

        ax.set_ylim(20, 0)
        ax.legend()
        return

    def save(
        self,
    ):
        raise NotImplementedError

def main():
    # First EDIT config_file at the beginning of this file to the location of config.toml
    # Instantiate a Lightcurves Object.
    lightcurves = Lightcurves(263.90117, 50.41108, name="AO Her")

    # Load provided lightcurves. EDIT THIS TO wherever you saved the data.
    lightcurve_folder = "../../data/rcb/AO Her"
    lightcurves.load_lightcurves_from_folder(lightcurve_folder)

    # At this point do whatever you want with the data.
    # Accessible through the format lightcurves.data[SURVEYNAME][FILTERNAME].OPTION
    # Replace capital letters with whatever you want, OPTION is ra, dec, time, magnitude, time_offset, colour and is consistent across surveys and filters.
    # When comparing different lightcurves add time_offset to time to change all to JD.

    # Example: Plot only Gaia and ATLAS data
    surveys_to_plot = ["gaia_dr3", "atlas"]

    # Make life easier by just using the data
    data = lightcurves.data

    # Setup matplotlib
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    ax = fig.add_subplot(111)

    ax.set_xlabel("Time (JD)")
    ax.set_ylabel("Magnitude")
    if not ax.yaxis_inverted():
        ax.invert_yaxis()

    ax.ticklabel_format(useOffset=False, style="plain")

    # Do plotting here, or any data analysis wanted.
    for my_survey in surveys_to_plot:
        for my_filter in data[my_survey]:
            label = f"{my_survey} {my_filter}"
            colour = tuple(color / 255 for color in data[my_survey][my_filter].colour)

            time = (
                data[my_survey][my_filter].time + data[my_survey][my_filter].time_offset
            )
            magnitude = data[my_survey][my_filter].magnitude

            ax.scatter(time, magnitude, color=colour, label=label)

    ax.set_title(
        f"Name: {lightcurves.name}, RA:{lightcurves.ra:.5f}, Dec:{lightcurves.dec:.5f}"
    )
    # Show and save plot.
    plt.legend()

    plt.savefig(f"{lightcurves.name}.png")
    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    main()