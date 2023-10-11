# -*- coding: utf-8 -*-
from ontomap.ontology.anatomy import MouseHumanOMDataset
from ontomap.ontology.biodiv import (
    EnvoSweetOMDataset, FishZooplanktonOMDataset,
    MacroalgaeMacrozoobenthosOMDataset,
    TaxrefldBacteriaNcbitaxonBacteriaOMDataset,
    TaxrefldChromistaNcbitaxonChromistaOMDataset,
    TaxrefldFungiNcbitaxonFungiOMDataset,
    TaxrefldPlantaeNcbitaxonPlantaeOMDataset,
    TaxrefldProtozoaNcbitaxonProtozoaOMDataset)
from ontomap.ontology.bioml import (NCITDOIDOMDataset, OMIMORDOOMDataset,
                                    SNOMEDFMABodyOMDataset,
                                    SNOMEDNCITNeoplasOMDataset,
                                    SNOMEDNCITPharmOMDataset)
from ontomap.ontology.commonkg import (NellDbpediaOMDataset,
                                       YagoWikidataOMDataset)
from ontomap.ontology.food import CiqualSirenOMDataset
from ontomap.ontology.phenotype import DoidOrdoOMDataset, HpMpOMDataset

ontology_matching = {
    "anatomy": [MouseHumanOMDataset],
    "food": [CiqualSirenOMDataset],
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
    "bioml": [
        NCITDOIDOMDataset,
        OMIMORDOOMDataset,
        SNOMEDFMABodyOMDataset,
        SNOMEDNCITNeoplasOMDataset,
        SNOMEDNCITPharmOMDataset,
    ],
}

__all__ = ["ontology_matching"]
