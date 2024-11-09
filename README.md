#GIS-702-Project

## Overview

This algorithm is designed to detect high rainfall events from large rainfall datasets. It includes multiple quality control checks to ensure data accuracy and reliability. The algorithm identifies high rainfall events and flags inconsistencies, enabling improved analysis of rainfall patterns and supporting meteorological research.

## File Structure

main.py: The entry point that orchestrates the QC workflow.
preprocessing.py: Loads and preprocesses the rainfall datasets.
logical.py: Tests logical consistency in the dataset.
completeness.py: Checks for dataset completeness.
temporal_quality.py: Assesses the temporal quality of data entries.
rainfall_analysis.py: Extracts high rainfall events.
storm_detection.py: Verify single-cell thunderstorms.
large_scale_rain.py: Verify large-scale rain events.

## Algorithm Workflow

Data Preprocessing: preprocessing.py loads datasets and performs initial cleaning.
Logical Consistency: logical.py ensures data adheres to logical expectations.
Completeness: completeness.py checks for missing or incomplete records.
Temporal Quality: temporal_quality.py assesses the continuity and regularity of data.
Rainfall Analysis: rainfall_analysis.py extracts instances of high rainfall for further inspection.
QC Checks:
Storm Detection: storm_detection.py identifies localized thunderstorms.
Large-Scale Rain Events: large_scale_rain.py flags regional rainfall events.

