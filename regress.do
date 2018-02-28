clear
insheet using "C:\Users\11316\Documents\0MCM\data\CA_B_norm.csv"
regress ng clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature

regress cl gdprx tetgr cltcd te ngmpb patcd hytcb
clear
insheet using "C:\Users\11316\Documents\0MCM\data\CA_B_norm.csv"
regress entropy clprb tpopp


clear
insheet using "C:\Users\11316\Documents\0MCM\data\AZ_B_norm.csv"
regress cl clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature

regress ng clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature
regress nu clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature
regress pm clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature
regress re clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature

regress cl GDPRX TPOPP TETGR TE HYTCB
regress ng gdprx tpopp tetgr te hytcb
regress nu gdprx tpopp tetgr te hytcb
regress pm gdprx tpopp tetgr te hytcb
regress re gdprx tpopp tetgr te hytcb










clear
insheet using "C:\Users\11316\Documents\0MCM\data\NM_B_norm.csv"
regress cl clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature

regress ng clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature
regress nu clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature
regress pm clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature
regress re clprb hytcb ngmpb paprb gdprx tetgr cltcd ngtcd patcd nuetd retcd estcd growthrate tpopp teacb teccb teicb teeib tercb precipitation temperature


regress cl ngtcd patcd te
regress ng ngtcd patcd te
regress pm ngtcd patcd te
regress re ngtcd patcd te




clear
insheet using "C:\Users\11316\Documents\0MCM\data\TX_B_norm.csv"
regress cl te gdprx cltcd patcd estcd tpopp
regress ng te gdprx cltcd patcd estcd tpopp
regress nu te gdprx cltcd patcd estcd tpopp
regress pm te gdprx cltcd patcd estcd tpopp
regress re te gdprx cltcd patcd estcd tpopp