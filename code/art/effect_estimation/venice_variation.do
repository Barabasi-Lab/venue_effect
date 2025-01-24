* Import solo dataset
import delimited "..\..\..\data\art\venice_matched.csv", clear

* Encode categorical variables and generate biennale year variable
encode gender_est, generate(gender_cat)
encode age_stage, generate(age_cat)
encode career_stage, generate(career_cat)
encode biennale_decade, generate(biennale_decade1)
gen new_year2biennale = year_diff + 5
xtset artist_id end_year

* Solo Analysis: Save ATET results
xthdidregress aipw (s f g b i.gender_cat europe asia northamerica southamerica oceania africa i.age_cat i.career_cat) (biennale_treated), group(artist_id)
estat aggregation, cohort graph

parmest, saving(venice_variation_solo_atet, replace)
use venice_variation_solo_atet.dta, clear
export delimited "..\..\..\results\art\venice_variation_solo_atet.csv", replace

* Import dataset again for group analysis
import delimited "..\..\..\data\art\venice_matched.csv", clear

* Encode categorical variables and generate biennale year variable
encode gender_est, generate(gender_cat)
encode age_stage, generate(age_cat)
encode career_stage, generate(career_cat)
encode biennale_decade, generate(biennale_decade1)
gen new_year2biennale = year_diff + 5
xtset artist_id end_year

* Group Analysis: Save ATET results
xthdidregress aipw (g f s b i.gender_cat europe asia northamerica southamerica oceania africa i.age_cat i.career_cat) (biennale_treated), group(artist_id)
estat aggregation, cohort graph

parmest, saving(venice_variation_group_atet, replace)
use venice_variation_group_atet.dta, clear
export delimited "..\..\..\results\art\venice_variation_group_atet.csv", replace

