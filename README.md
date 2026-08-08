# skywatch-aq
Satellite-based air quality health alerts for cities without ground monitors — built with IBM Bob for the AI Builders Challenge

## Project Status: In development

## Architecture

NASA MODIS/VIIRS satellite Aerosol Optical Depth (AOD) data is downloaded via the Earthdata API and combined with ground-level PM2.5 readings from OpenAQ monitoring stations. A machine learning model (scikit-learn) is trained to map AOD and meteorological features to surface PM2.5 concentrations. The trained model is then applied to locations that have satellite coverage but no ground monitors, producing PM2.5 estimates that are fed into a rule-based health alert engine — delivering plain-language air quality advisories via a Streamlit web app.

## Built with IBM Bob

IBM Bob (AI Builders Challenge) is the primary development tool for this project — from scaffolding and data pipelines through model training and the Streamlit front-end.
