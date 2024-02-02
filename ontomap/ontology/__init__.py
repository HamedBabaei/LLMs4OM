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
    NCITDOIDDiseaseOMDataset,
    OMIMORDODiseaseOMDataset,
    SNOMEDFMABodyOMDataset,
    SNOMEDNCITNeoplasOMDataset,
    SNOMEDNCITPharmOMDataset,
)
from ontomap.ontology.commonkg import NellDbpediaOMDataset, YagoWikidataOMDataset
from ontomap.ontology.mse import (
    MaterialInformationEMMOOMDataset,
    MaterialInformationMatOntoMDataset,
)
from ontomap.ontology.phenotype import DoidOrdoOMDataset, HpMpOMDataset

ontology_matching = {
    "anatomy": [MouseHumanOMDataset],
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
    "mse": [
        MaterialInformationEMMOOMDataset,
        MaterialInformationMatOntoMDataset
    ],
}

__all__ = ["ontology_matching"]
