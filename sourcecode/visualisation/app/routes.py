# -*- coding: UTF-8 -*-
from app import app
from flask import request
from flask import jsonify
from flask import render_template
import sys, json

# This folder and python file/function must be present in order for the visualisation app to run.
# Please see the readme for more details.
sys.path.append('../candidate_extraction')
from triples_from_text import extract_triples

sys.path.append('app/visualisation_system')
from add_visualisation_information import *


@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html', title='Home')


# Convert a list of strings (documents) into a single-dimensional list of triples.
def docs2triples(documents):
	all_triples = []
	for doc_idx, d in enumerate(documents):
		triples = extract_triples(d)	 # Extract triples using our triple extraction system
						 # (Note: this could be replaced by a different model if desired)
		triples = add_visualisation_information(triples, d)	# Add the visualisation information to each triple, i.e. degree, betweenness, SemEval relation, NER type.
		for t in triples:
			all_triples.append([doc_idx] + t)	# Prepend the document index to each triple.
	return all_triples

# Take a list of documents from the client and return a list of triples.
@app.route('/text2kg/get_triples', methods=['POST'])
def get_triples():	
	documents = request.get_json(force=True)['documents']
	print(">>>", documents)
	all_triples = docs2triples(documents)	
	print(">>>", all_triples)
	return jsonify(all_triples)


# Load the initial demo graph, and save the data to javascript files.
# This ensures that the initial graph loads almost instantly on first page load.
def load_initial_data():
	documents = ["\"\"Make it more confident and intuitive to drive."" That was the feedback current owners of Ford's wild-child, track-rat, flat-plane-crank, manual-only Mustang Shelby GT350 offered when the development team went asking. Although buyers of the Shelby GT350R tend to be pretty accomplished shoes, buyers of the ""base"" GT350 are more likely to drive their cars daily and take them to a track much less frequently. As such, these owners tend to be less practiced and more fearful of wadding up their babies. The team kept this wish in mind as it pushed the current Mustang platform to new levels of performance while developing the forthcoming GT500, and is now rolling out a new GT500 with higher performance limits that are easier to reach. (The GT350R remains unchanged for 2019.) Assisting with the aforementioned development was veteran race driver Billy Johnson, who's spent three years working with the team while also racing Ford GTs and prepping the Mustang GT4 race car, which he's campaigning this year.", "Key PointsFord Motor issued a recall of more than 270,000 Fusion vehicles in North America because of a transmission issue that can allow the cars to shift into a different gear than the one the driver selected.This can cause the vehicles to roll away if the parking brake is not applied.Ford said it is aware of three reports of property damage due to the issue and one injury  ""potentially"" related to the problem.Ford Motor Co. Fusion vehicles move down the production line at the Flat Rock Assembly Plant in Flat Rock, Michigan.Jeff Kowalsky | Bloomberg | Getty ImagesFord Motor said Wednesday that it is recalling more than 270,000 Fusion vehicles in North America to fix a transmission glitch that can cause the car to shift gears and roll away.The recall is for 2013-16 Fusion vehicles with 2.5-liter engines that were built at the automaker's Flat Rock, Michigan, and Hermosillo, Mexico, assembly plants.The company said the bushing that attaches the shifter cable to the vehicle's transmission may detach, which can result in  ""unintended vehicle movement.\"\"", "The Puma is coming back, but not as a stylish coupe. You could say April was the month of SUVs at Ford taking into consideration that aside from unveiling the next-generation Kuga / Escape and the Europe-bound Explorer PHEV, the Blue Oval also previewed a reborn Puma. Well, maybe “reborn” is not the best term to use since it won’t be coming back as the attractive coupe it was during the late 1990s and early 2000s, but as a small SUV to complement the EcoSport. During the event held in Amsterdam, Ford brought the all-new Puma on stage, but dimmed the lights to conceal the SUV’s design. For this Friday’s render, we asked our Photoshop master to figure out what hiding in the darkness by imagining the 2020 Puma in production guise. Since it’s essentially an SUV version of the latest-generation, not-for-America Fiesta, it makes sense that the styling is heavily inspired by the supermini."]
	all_triples = docs2triples(documents)
	with open("../visualisation/app/static/javascript/initial_triples.js", "w") as f:
		f.write("initial_triples = ")
		f.write(json.dumps(all_triples))
	with open("../visualisation/app/static/javascript/initial_documents.js", "w") as f:
		f.write("initial_documents = ")
		f.write(json.dumps(documents))

load_initial_data()
