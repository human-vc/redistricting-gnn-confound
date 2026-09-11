from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import geopandas as gpd
import maup


class GraphRepairError(RuntimeError):
    pass


@dataclass(frozen=True)
class StateConfig:
    name: str
    postal: str
    shapefile_path: str
    native_crs_epsg: int
    n_districts: int
    pop_col: str
    enacted_plan_col: str
    enacted_plan_case: str
    election_dem_col: str
    election_rep_col: str
    election_label: str
    vap_total_col: str
    vap_black_col: str
    vap_hisp_col: str
    vap_white_col: str
    cols_to_add: list[str] = field(default_factory=list)
    enacted_plan_fix_edges: tuple[tuple[int, int], ...] = ()


PA_CONFIG = StateConfig(
    name="Pennsylvania",
    postal="PA",
    shapefile_path="data/raw/PA/PA.shp",
    native_crs_epsg=26918,
    n_districts=18,
    pop_col="TOTPOP",
    enacted_plan_col="CD_2011",
    enacted_plan_case="League of Women Voters v. Pennsylvania, 178 A.3d 737 (Pa. 2018)",
    election_dem_col="T16PRESD",
    election_rep_col="T16PRESR",
    election_label="2016 President",
    vap_total_col="VAP",
    vap_black_col="BVAP",
    vap_hisp_col="HVAP",
    vap_white_col="WVAP",
    cols_to_add=["TOTPOP", "VAP", "BVAP", "HVAP", "WVAP", "T16PRESD", "T16PRESR", "T16PRESOTH", "CD_2011"],
    enacted_plan_fix_edges=((7648, 7635), (1247, 1160)),
)

NC_CONFIG = StateConfig(
    name="North Carolina",
    postal="NC",
    shapefile_path="data/raw/NC/NC_VTD.shp",
    native_crs_epsg=6543,
    n_districts=13,
    pop_col="TOTPOP",
    enacted_plan_col="newplan",
    enacted_plan_case="Common Cause v. Rucho, 318 F. Supp. 3d 777 (M.D.N.C. 2018)",
    election_dem_col="EL16G_PR_D",
    election_rep_col="EL16G_PR_R",
    election_label="2016 President",
    vap_total_col="VAP",
    vap_black_col="BVAP",
    vap_hisp_col="HVAP",
    vap_white_col="WVAP",
    cols_to_add=["TOTPOP", "VAP", "BVAP", "HVAP", "WVAP", "EL16G_PR_D", "EL16G_PR_R", "newplan"],
)

MD_CONFIG = StateConfig(
    name="Maryland",
    postal="MD",
    shapefile_path="data/raw/MD/MD-precincts.shp",
    native_crs_epsg=26985,
    n_districts=8,
    pop_col="TOTPOP",
    enacted_plan_col="CD",
    enacted_plan_case="Lamone v. Benisek / Rucho v. Common Cause, 588 U.S. 684 (2019); MD-06",
    election_dem_col="PRES16D",
    election_rep_col="PRES16R",
    election_label="2016 President",
    vap_total_col="VAP",
    vap_black_col="BVAP",
    vap_hisp_col="HVAP",
    vap_white_col="WVAP",
    cols_to_add=["TOTPOP", "VAP", "BVAP", "HVAP", "WVAP", "PRES16D", "PRES16R", "CD"],
)

STATE_CONFIGS = {"PA": PA_CONFIG, "NC": NC_CONFIG, "MD": MD_CONFIG}


def load_repaired_geodataframe(config: StateConfig, root: str) -> gpd.GeoDataFrame:
    path = os.path.join(root, config.shapefile_path)
    gdf = gpd.read_file(path)
    if gdf.crs is None:
        raise GraphRepairError(f"{config.postal}: shapefile has no CRS attached ({path}).")
    if gdf.crs.to_epsg() != config.native_crs_epsg:
        warnings.warn(
            f"{config.postal}: expected native CRS EPSG:{config.native_crs_epsg}, got EPSG:{gdf.crs.to_epsg()}.",
            stacklevel=2,
        )
    gdf = gdf.to_crs(gdf.estimate_utm_crs())
    if not maup.doctor(gdf, silent=True):
        gdf = gdf.copy()
        gdf["geometry"] = maup.smart_repair(gdf["geometry"])
        if not maup.doctor(gdf, silent=True):
            raise GraphRepairError(f"{config.postal}: maup.smart_repair() did not resolve topology issues.")
    return gdf
