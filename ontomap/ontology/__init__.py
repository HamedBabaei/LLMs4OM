# -*- coding: utf-8 -*-
from ontomap.ontology.anatomy import MouseHumanOMDataset
from ontomap.ontology.biodiv import (
    EnvoSweetOMDataset,
    FishZooplanktonOMDataset,
    MacroalgaeMacrozoobenthosOMDataset,
    TaxrefldBacteriaNcbitaxonBacteriaOMDataset,
    TaxrefldChromistaNcbitaxonChromistaOMDataset,
    TaxrefldFungiNcbitaxonFungiOMDataset,
    TaxrefldPlantaeNcbitaxonPlantaeOMDataset,
    TaxrefldProtozoaNcbitaxonProtozoaOMDataset,
)
from ontomap.ontology.bioml import (
    NCITDOIDDiseaseLLMOMDataset,
    NCITDOIDDiseaseOMDataset,
    OMIMORDODiseaseOMDataset,
    SNOMEDFMABodyLLMOMDataset,
    SNOMEDFMABodyOMDataset,
    SNOMEDNCITNeoplasOMDataset,
    SNOMEDNCITPharmOMDataset,
)
from ontomap.ontology.commonkg import NellDbpediaOMDataset, YagoWikidataOMDataset
from ontomap.ontology.food import CiqualSirenOMDataset
from ontomap.ontology.mse import (
    MaterialInformationEMMOOMDataset,
    MaterialInformationMatOntoMDataset,
    MaterialInformationMatOntoReducedMDataset,
)
from ontomap.ontology.phenotype import DoidOrdoOMDataset, HpMpOMDataset

ontology_matching = {
    "anatomy": [MouseHumanOMDataset],
    # "food": [CiqualSirenOMDataset],
    "biodiv": [
        EnvoSweetOMDataset,
        FishZooplanktonOMDataset,
        MacroalgaeMacrozoobenthosOMDataset,
        TaxrefldBacteriaNcbitaxonBacteriaOMDataset,
        TaxrefldChromistaNcbitaxonChromistaOMDataset,
        TaxrefldFungiNcbitaxonFungiOMDataset,
        TaxrefldPlantaeNcbitaxonPlantaeOMDataset,
        TaxrefldProtozoaNcbitaxonProtozoaOMDataset,
    ],
    "phenotype": [DoidOrdoOMDataset, HpMpOMDataset],
    "commonkg": [NellDbpediaOMDataset, YagoWikidataOMDataset],
    "bio-ml": [
        NCITDOIDDiseaseOMDataset,
        OMIMORDODiseaseOMDataset,
        SNOMEDFMABodyOMDataset,
        SNOMEDNCITNeoplasOMDataset,
        SNOMEDNCITPharmOMDataset,
    ],
    # "bio-llm": [SNOMEDFMABodyLLMOMDataset, NCITDOIDDiseaseLLMOMDataset],
    "mse": [
        MaterialInformationEMMOOMDataset,
        MaterialInformationMatOntoMDataset,
        MaterialInformationMatOntoReducedMDataset,
    ],
}

__all__ = ["ontology_matching"]
