
from collections import namedtuple

Annotation = namedtuple('Annotation', ['anchor_text', 'anchor_text_location', 'entity_id', 'entity_name'])
AnchorTextLocation = namedtuple('AnchorTextLocation', ['start', 'end'])

annotation_dict = {}

# Use AnnotationNotebook to help!

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# 'enwiki:Aftertaste' annotations
annotation_dict['enwiki:Aftertaste'] = [
    # ===========================================================
    # ========================= INDEX 0 =========================
    # ===========================================================
    # Aftertaste is the taste intensity of a food or beverage that is perceived immediately after that food or beverage
    # is removed from the mouth. The aftertastes of different foods and beverages can vary by intensity and over time,
    # but the unifying feature of aftertaste is that it is perceived after a food or beverage is either swallowed or
    # spat out. The neurobiological mechanisms of taste (and aftertaste) signal transduction from the taste receptors
    # in the mouth to the brain have not yet been fully understood. However, the primary taste processing area located
    # in the insula has been observed to be involved in aftertaste perception.
    [
        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=18, end=23),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=39, end=43),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='beverage', anchor_text_location=AnchorTextLocation(start=47, end=55),
                   entity_id='enwiki:Drink', entity_name='Drink'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=97, end=101),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='beverage', anchor_text_location=AnchorTextLocation(start=105, end=113),
                   entity_id='enwiki:Drink', entity_name='Drink'),

        Annotation(anchor_text='foods', anchor_text_location=AnchorTextLocation(start=170, end=175),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='beverages', anchor_text_location=AnchorTextLocation(start=180, end=189),
                   entity_id='enwiki:Drink', entity_name='Drink'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=298, end=302),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='beverage', anchor_text_location=AnchorTextLocation(start=306, end=314),
                   entity_id='enwiki:Drink', entity_name='Drink'),

        Annotation(anchor_text='neurobiological', anchor_text_location=AnchorTextLocation(start=352, end=367),
                   entity_id='enwiki:Neuroscience', entity_name='Neuroscience'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=382, end=387),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='signal transduction', anchor_text_location=AnchorTextLocation(start=405, end=424),
                   entity_id='enwiki:Signal%20transduction', entity_name='Signal transduction'),

        Annotation(anchor_text='taste receptors', anchor_text_location=AnchorTextLocation(start=434, end=449),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='insula', anchor_text_location=AnchorTextLocation(start=570, end=576),
                   entity_id='enwiki:Insular%20cortex', entity_name='Insular cortex'),

        Annotation(anchor_text='taste processing area', anchor_text_location=AnchorTextLocation(start=533, end=554),
                   entity_id='enwiki:Taste', entity_name='Taste'),
    ],
    # ===========================================================
    # ========================= INDEX 1 =========================
    # ===========================================================
    # Characteristics of a food's aftertaste are quality, intensity, and duration. Quality describes the actual taste of
    # a food and intensity conveys the magnitude of that taste. Duration describes how long a food's aftertaste
    # sensation lasts. Foods that have lingering aftertastes typically have long sensation durations.
    [
        Annotation(anchor_text='food\'s', anchor_text_location=AnchorTextLocation(start=21, end=27),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=106, end=111),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=117, end=121),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=166, end=171),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text="food's", anchor_text_location=AnchorTextLocation(start=203, end=209),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text="Foods", anchor_text_location=AnchorTextLocation(start=238, end=243),
                   entity_id='enwiki:Food', entity_name='Food'),
    ],
    # ===========================================================
    # ========================= INDEX 2 =========================
    # ===========================================================
    # Because taste perception is unique to every person, descriptors for taste quality and intensity have been
    # standardized, particularly for use in scientific studies. For taste quality, foods can be described by the
    # commonly used terms "sweet", "sour", "salty", "bitter", "umami", or "no taste". Description of aftertaste
    # perception relies heavily upon the use of these words to convey the taste that is being sensed after a food has
    # been removed from the mouth.

    [
        Annotation(anchor_text='taste perception', anchor_text_location=AnchorTextLocation(start=8, end=24),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='taste quality and intensity',
                   anchor_text_location=AnchorTextLocation(start=68, end=95), entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='taste quality', anchor_text_location=AnchorTextLocation(start=168, end=181),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text="foods", anchor_text_location=AnchorTextLocation(start=183, end=188),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=387, end=392),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text="food", anchor_text_location=AnchorTextLocation(start=422, end=426),
                   entity_id='enwiki:Food', entity_name='Food'),

    ],
    # ===========================================================
    # ========================= INDEX 3 =========================
    # ===========================================================
    # The description of taste intensity is also subject to variability among individuals. Variations of the Borg
    # Category Ratio Scale or other similar metrics are often used to assess the intensities of foods. The scales
    # typically have categories that range from either zero or one through ten (or sometimes beyond ten) that describe
    # the taste intensity of a food. A score of zero or one would correspond to unnoticeable or weak taste intensities,
    # while a higher score would correspond to moderate or strong taste intensities. It is the prolonged moderate or
    # strong taste intensities that persist even after a food is no longer present in the mouth that describe aftertaste
    # sensation.

    [
        Annotation(anchor_text='taste intensity', anchor_text_location=AnchorTextLocation(start=19, end=34),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='Borg Category Ratio Scale', anchor_text_location=AnchorTextLocation(start=103, end=128),
                   entity_id='enwiki:Rating%20of%20perceived%20exertion', entity_name='Rating of perceived exertion'),

        Annotation(anchor_text="foods", anchor_text_location=AnchorTextLocation(start=198, end=203),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste intensity', anchor_text_location=AnchorTextLocation(start=333, end=348),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text="food", anchor_text_location=AnchorTextLocation(start=354, end=358),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste intensities', anchor_text_location=AnchorTextLocation(start=424, end=441),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='taste intensities', anchor_text_location=AnchorTextLocation(start=503, end=520),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='taste intensities', anchor_text_location=AnchorTextLocation(start=561, end=578),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text="food", anchor_text_location=AnchorTextLocation(start=605, end=609),
                   entity_id='enwiki:Food', entity_name='Food'),
    ],
    # ===========================================================
    # ========================= INDEX 4 =========================
    # ===========================================================
    # Foods that have distinct aftertastes are distinguished by their temporal profiles, or how long their tastes are
    # perceived during and after consumption. A sample testing procedure to measure a food's temporal profile would
    # entail first recording the time of onset for initial taste perception when the food is consumed, and then
    # recording the time at which there is no longer any perceived taste. The difference between these two values
    # yields the total time of taste perception. Match this with intensity assessments over the same time interval and
    # a representation of the food's taste intensity over time can be obtained. With respect to aftertaste, this type
    # of testing would have to measure the onset of taste perception from the point after which the food was removed
    # from the mouth.
    [
        Annotation(anchor_text="Foods", anchor_text_location=AnchorTextLocation(start=0, end=5),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='tastes', anchor_text_location=AnchorTextLocation(start=101, end=107),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food\'s', anchor_text_location=AnchorTextLocation(start=192, end=198),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste perception', anchor_text_location=AnchorTextLocation(start=275, end=291),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=301, end=305),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=389, end=394),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='total time of taste perception', anchor_text_location=AnchorTextLocation(start=447, end=477),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food\'s', anchor_text_location=AnchorTextLocation(start=573, end=579),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste intensity over time', anchor_text_location=AnchorTextLocation(start=580, end=605),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='onset of taste perception', anchor_text_location=AnchorTextLocation(start=698, end=723),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=755, end=759),
                   entity_id='enwiki:Food', entity_name='Food'),
    ],
    # ===========================================================
    # ========================= INDEX 5 =========================
    # ===========================================================
    # The categorization of people into "tasters" or "nontasters" based on their sensitivity to the bitterness of
    # propylthiouracil and the expression of fungiform papillae on their tongues has suggested that the variations from
    # person-to-person observed in taste perception are genetically based. If so, then sensations of aftertaste could
    # also be affected by the activities of specific genes that affect an individual's perception of different foods.
    # For example, the intensity of the aftertaste sensations "nontasters" experienced after caffeine consumption was
    # found to diminish faster than the sensations "tasters" experienced. This may imply that because of their taste
    # bud profiles, "tasters" may be more sensitive to the tastes of different foods, and thus experience a more
    # persistent sensation of those foods' tastes.
    [

        Annotation(anchor_text='tasters', anchor_text_location=AnchorTextLocation(start=35, end=42),
                   entity_id='enwiki:Supertaster', entity_name='Supertaster'),

        Annotation(anchor_text='propylthiouracil', anchor_text_location=AnchorTextLocation(start=108, end=124),
                   entity_id='enwiki:Propylthiouracil', entity_name='Propylthiouracil'),

        Annotation(anchor_text='fungiform papillae', anchor_text_location=AnchorTextLocation(start=147, end=165),
                   entity_id='enwiki:Lingual%20papillae', entity_name='Lingual papillae'),

        Annotation(anchor_text='taste perception', anchor_text_location=AnchorTextLocation(start=251, end=267),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='foods', anchor_text_location=AnchorTextLocation(start=439, end=444),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='tasters', anchor_text_location=AnchorTextLocation(start=604, end=611),
                   entity_id='enwiki:Supertaster', entity_name='Supertaster'),

        Annotation(anchor_text='tasters', anchor_text_location=AnchorTextLocation(start=684, end=691),
                   entity_id='enwiki:Supertaster', entity_name='Supertaster'),

        Annotation(anchor_text='tastes', anchor_text_location=AnchorTextLocation(start=722, end=728),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='foods', anchor_text_location=AnchorTextLocation(start=742, end=747),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='foods\'', anchor_text_location=AnchorTextLocation(start=806, end=812),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='tastes', anchor_text_location=AnchorTextLocation(start=813, end=819),
                   entity_id='enwiki:Taste', entity_name='Taste'),
    ],
    # ===========================================================
    # ========================= INDEX 6 =========================
    # ===========================================================
    # Because a lingering taste sensation is intrinsic to aftertaste, the molecular mechanisms that underlie aftertaste
    # are presumed to be linked to either the continued or delayed activation of receptors and signaling pathways in
    # the mouth that are involved in taste processing. The current understanding of how a food's taste is communicated
    # to the brain is as follows:
    [
        Annotation(anchor_text='taste sensation', anchor_text_location=AnchorTextLocation(start=20, end=35),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='receptors', anchor_text_location=AnchorTextLocation(start=189, end=198),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='taste processing', anchor_text_location=AnchorTextLocation(start=256, end=272),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food\'s', anchor_text_location=AnchorTextLocation(start=309, end=315),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=316, end=321),
                   entity_id='enwiki:Taste', entity_name='Taste'),
    ],
    # ===========================================================
    # ========================= INDEX 7 =========================
    # ===========================================================
    # Chemicals in food interact with receptors on the taste receptor cells located on the tongue and the roof of the
    # mouth. These interactions can be affected by temporal and spatial factors like the time of receptor activation or
    # the particular taste receptors that are activated (sweet, salty, bitter, etc.).
    [
        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=13, end=17),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='receptors', anchor_text_location=AnchorTextLocation(start=32, end=41),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='taste receptor cells', anchor_text_location=AnchorTextLocation(start=49, end=69),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='receptor', anchor_text_location=AnchorTextLocation(start=203, end=211),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='taste receptors', anchor_text_location=AnchorTextLocation(start=241, end=256),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor')
    ],
    # ===========================================================
    # ========================= INDEX 8 =========================
    # ===========================================================
    # The chorda tympani (cranial nerve VII), the glossopharyngeal nerve (cranial nerve IX), and the vagus nerve
    # (cranial nerve X) carry information from the taste receptors to the brain for cortical processing.
    [
        Annotation(anchor_text='chorda tympani', anchor_text_location=AnchorTextLocation(start=4, end=18),
                   entity_id='enwiki:Chorda%20tympani', entity_name='Chorda tympani'),

        Annotation(anchor_text='glossopharyngeal nerve', anchor_text_location=AnchorTextLocation(start=44, end=66),
                   entity_id='enwiki:Glossopharyngeal%20nerve', entity_name='Glossopharyngeal nerve'),

        Annotation(anchor_text='vagus nerve', anchor_text_location=AnchorTextLocation(start=95, end=106),
                   entity_id='enwiki:Vagus%20nerve', entity_name='Vagus nerve'),

        Annotation(anchor_text='taste receptors', anchor_text_location=AnchorTextLocation(start=152, end=167),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor')
    ],
    # ===========================================================
    # ========================= INDEX 9 =========================
    # ===========================================================
    # In the context of aftertaste, the combination of both receptor-dependent and receptor-independent processes have
    # been proposed to explain the signal transduction mechanisms for foods with distinct aftertastes, particularly
    # those that are bitter. The receptor-dependent process is the same as what was described above. However, the
    # receptor-independent process involves the diffusion of bitter, amphiphilic chemicals like quinine across the
    # taste receptor cell membranes. Once inside the taste receptor cell, these compounds have been observed to
    # activate intracellular G-proteins and other proteins that are involved in signaling pathways routed to the brain.
    # The bitter compounds thus activate both the taste receptors on the cell surface, as well as the signaling pathway
    # proteins in the intracellular space. Intracellular signaling may be slower than taste cell receptor activation
    # since more time is necessary for the bitter compounds to diffuse across the cell membrane and interact with
    # intracellular proteins. This delayed activation of intracellular signaling proteins in response to the bitter
    # compounds, in addition to the extracellular receptor signaling is proposed to be related to the lingering
    # aftertaste associated with bitter foods. The combination of both mechanisms leads to an overall longer response
    # of the taste receptor cells to the bitter foods, and aftertaste perception subsequently occurs.
    [
        Annotation(anchor_text='signal transduction', anchor_text_location=AnchorTextLocation(start=142, end=161),
                   entity_id='enwiki:Signal%20transduction', entity_name='Signal transduction'),

        Annotation(anchor_text='foods', anchor_text_location=AnchorTextLocation(start=177, end=182),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='amphiphilic', anchor_text_location=AnchorTextLocation(start=394, end=405),
                   entity_id='enwiki:Amphiphile', entity_name='Amphiphile'),

        Annotation(anchor_text='taste receptor cell membranes', anchor_text_location=AnchorTextLocation(start=440, end=469),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='taste receptor cell',anchor_text_location=AnchorTextLocation(start=487, end=506),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='these compounds', anchor_text_location=AnchorTextLocation(start=508, end=523),
                   entity_id='enwiki:Amphiphile', entity_name='Amphiphile'),

        Annotation(anchor_text='quinine', anchor_text_location=AnchorTextLocation(start=421, end=428),
                   entity_id='enwiki:Quinine', entity_name='Quinine'),

        Annotation(anchor_text='G-proteins', anchor_text_location=AnchorTextLocation(start=569, end=579),
                   entity_id='enwiki:G%20protein', entity_name='G protein'),

        Annotation(anchor_text='taste receptors', anchor_text_location=AnchorTextLocation(start=704, end=719),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='taste cell receptor', anchor_text_location=AnchorTextLocation(start=854, end=873),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='foods', anchor_text_location=AnchorTextLocation(start=1243, end=1248),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste receptor cells', anchor_text_location=AnchorTextLocation(start=1328, end=1348),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='foods', anchor_text_location=AnchorTextLocation(start=1363, end=1368),
                   entity_id='enwiki:Food', entity_name='Food'),
    ],
    # ===========================================================
    # ========================= INDEX 10 ========================
    # ===========================================================
    # The primary taste perception areas in the cerebral cortex are located in the insula and regions of the
    # somatosensory cortex; the nucleus of the solitary tract located in the brainstem also plays a major role in taste
    # perception. These regions were identified when human subjects were exposed to a taste stimulus and their cerebral
    # blood flow measured with magnetic resonance imaging. Although these regions have been identified as the primary
    # zones for taste processing in the brain, other cortical areas are also activated during eating, as other sensory
    # inputs are being signaled to the cortex.
    [
        Annotation(anchor_text='taste perception', anchor_text_location=AnchorTextLocation(start=12, end=28),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='insula', anchor_text_location=AnchorTextLocation(start=77, end=83),
                   entity_id='enwiki:Insular%20cortex', entity_name='Insular cortex'),

        Annotation(anchor_text='somatosensory cortex', anchor_text_location=AnchorTextLocation(start=103, end=123),
                   entity_id='enwiki:Postcentral%20gyrus', entity_name='Postcentral gyrus'),

        Annotation(anchor_text='nucleus of the solitary tract', anchor_text_location=AnchorTextLocation(start=129, end=158),
                   entity_id='enwiki:Solitary%20nucleus', entity_name='Solitary nucleus'),

        Annotation(anchor_text='taste perception', anchor_text_location=AnchorTextLocation(start=211, end=227),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='taste stimulus', anchor_text_location=AnchorTextLocation(start=297, end=311),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='taste processing', anchor_text_location=AnchorTextLocation(start=453, end=469),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='cortex', anchor_text_location=AnchorTextLocation(start=589, end=595),
                   entity_id='enwiki:Postcentral%20gyrus', entity_name='Postcentral gyrus'),

        Annotation(anchor_text='magnetic resonance imaging', anchor_text_location=AnchorTextLocation(start=356, end=382),
                   entity_id='enwiki:Magnetic%20resonance%20imaging', entity_name='Magnetic resonance imaging'),

    ],
    # ===========================================================
    # ========================= INDEX 11 ========================
    # ===========================================================
    # For aftertaste, much is unclear about the cortical processing related to its perception. The first neuroimaging
    # study to evaluate the temporal taste profile of aspartame, an artificial sweetener, in humans was published in
    # 2009. In it, the insula was observed to be activated for a longer period of time than other sensory processing
    # areas in the brain when the aftertaste profile of aspartame was measured. Subjects were administered a solution
    # of aspartame for a specific amount of time before being instructed to swallow the solution. Functional magnetic
    # resonance images of the blood flow in the subjects' brains were recorded before and after they swallowed the
    # aspartame solution. Before swallowing, the amygdala, somatosensory cortex, thalamus, and basal ganglia were all
    # activated. After swallowing, only the insula remained activated and the response of the other brain regions was
    # not evident. This suggests that the insula may be a primary region for aftertaste sensation because it was
    # activated even after the aspartame solution was no longer present in the mouth. This finding aligns with the
    # insula's identification as a central taste processing area and simply expands its function. An explanation for
    # less activation of the amygdala was that because it is a reward center in the brain, less reward would be
    # experienced by the subjects during prolonged exposure to the aspartame solution.
    [
        Annotation(anchor_text='taste profile', anchor_text_location=AnchorTextLocation(start=143, end=156),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='aspartame', anchor_text_location=AnchorTextLocation(start=160, end=169),
                   entity_id='enwiki:Aspartame', entity_name='Aspartame'),

        Annotation(anchor_text='insula', anchor_text_location=AnchorTextLocation(start=240, end=246),
                   entity_id='enwiki:Insular%20cortex', entity_name='Insular cortex'),

        Annotation(anchor_text='aspartame', anchor_text_location=AnchorTextLocation(start=384, end=393),
                   entity_id='enwiki:Aspartame', entity_name='Aspartame'),

        Annotation(anchor_text='aspartame', anchor_text_location=AnchorTextLocation(start=449, end=458),
                   entity_id='enwiki:Aspartame', entity_name='Aspartame'),

        Annotation(anchor_text='aspartame solution', anchor_text_location=AnchorTextLocation(start=667, end=685),
                   entity_id='enwiki:Aspartame', entity_name='Aspartame'),

        Annotation(anchor_text='amygdala', anchor_text_location=AnchorTextLocation(start=710, end=718),
                   entity_id='enwiki:Amygdala', entity_name='Amygdala'),

        Annotation(anchor_text='somatosensory cortex', anchor_text_location=AnchorTextLocation(start=720, end=740),
                   entity_id='enwiki:Postcentral%20gyrus', entity_name='Postcentral gyrus'),

        Annotation(anchor_text='thalamus', anchor_text_location=AnchorTextLocation(start=742, end=750),
                   entity_id='enwiki:Thalamus', entity_name='Thalamus'),

        Annotation(anchor_text='basal ganglia', anchor_text_location=AnchorTextLocation(start=756, end=769),
                   entity_id='enwiki:Basal%20ganglia', entity_name='Basal ganglia'),

        Annotation(anchor_text='insula', anchor_text_location=AnchorTextLocation(start=817, end=823),
                   entity_id='enwiki:Insular%20cortex', entity_name='Insular cortex'),

        Annotation(anchor_text='insula', anchor_text_location=AnchorTextLocation(start=927, end=933),
                   entity_id='enwiki:Insular%20cortex', entity_name='Insular cortex'),

        Annotation(anchor_text='aspartame solution', anchor_text_location=AnchorTextLocation(start=1023, end=1041),
                   entity_id='enwiki:Aspartame', entity_name='Aspartame'),

        Annotation(anchor_text='insula\'s', anchor_text_location=AnchorTextLocation(start=1107, end=1115),
                   entity_id='enwiki:Insular%20cortex', entity_name='Insular cortex'),

        Annotation(anchor_text='taste processing', anchor_text_location=AnchorTextLocation(start=1144, end=1160),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='amygdala', anchor_text_location=AnchorTextLocation(start=1241, end=1249),
                   entity_id='enwiki:Amygdala', entity_name='Amygdala'),

        Annotation(anchor_text='aspartame solution', anchor_text_location=AnchorTextLocation(start=1385, end=1403),
                   entity_id='enwiki:Aspartame', entity_name='Aspartame'),
    ],
    # ===========================================================
    # ========================= INDEX 12 ========================
    # ===========================================================
    # Flavor is an emergent property that is the combination of multiple sensory systems including olfaction, taste,
    # and somatosensation. How the flavor of a food is perceived, whether it is unpleasant or satisfying, is stored as
    # a memory so that the next time the same (or a similar) food is encountered, the previous experience can be
    # recalled and a decision made to consume that food. This process of multisensory inputs to the brain during
    # eating, followed by learning from eating experiences is the central idea of flavor processing.Richard Stevenson
    # mentions in The Psychology of Flavour that people often do not realize that a food's flavor can be described by
    # the food's smell, taste, or texture. Instead, he claims, people perceive flavor as a "unitary percept", in which
    # a descriptor for either taste or smell is used to describe a food's flavor. Consider the terms that are used to
    # describe the flavors of foods. For instance, a food may taste sweet, but often its flavor is described as such
    # while not considering its smell or other sensory characteristics. For example, honey tastes sweet so its smell
    # is associated with that descriptor, and sweet is also used to describe its flavor. In fact, sweetness is one of
    # the four basic taste qualities and only comprises part of a food's flavor.
    [
        Annotation(anchor_text='Flavor', anchor_text_location=AnchorTextLocation(start=0, end=6),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='olfaction', anchor_text_location=AnchorTextLocation(start=93, end=102),
                   entity_id='enwiki:Olfaction', entity_name='Olfaction'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=104, end=109),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='somatosensation', anchor_text_location=AnchorTextLocation(start=115, end=130),
                   entity_id='enwiki:Somatosensory%20system', entity_name='Somatosensory system'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=140, end=146),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=152, end=156),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=279, end=283),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=376, end=380),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='flavor processing', anchor_text_location=AnchorTextLocation(start=514, end=531),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='food\'s', anchor_text_location=AnchorTextLocation(start=628, end=634),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=635, end=641),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='food\'s', anchor_text_location=AnchorTextLocation(start=666, end=672),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=680, end=685),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=735, end=741),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=799, end=804),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food\'s', anchor_text_location=AnchorTextLocation(start=836, end=842),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=843, end=849),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='flavors', anchor_text_location=AnchorTextLocation(start=900, end=907),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='foods', anchor_text_location=AnchorTextLocation(start=911, end=916),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='food', anchor_text_location=AnchorTextLocation(start=934, end=938),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=943, end=948),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=970, end=976),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='tastes', anchor_text_location=AnchorTextLocation(start=1083, end=1089),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=1184, end=1190),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='taste qualities', anchor_text_location=AnchorTextLocation(start=1236, end=1251),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='food\'s', anchor_text_location=AnchorTextLocation(start=1281, end=1287),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=1288, end=1294),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),
    ],
    # ===========================================================
    # ========================= INDEX 13 ========================
    # ===========================================================
    # Unlike flavor, aftertaste is a solely gustatory event that is not considered to involve any of the other major
    # senses. The distinction of being based on one (aftertaste) versus multiple (flavor) sensory inputs is what
    # separates the two phenomena.
    [
        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=7, end=13),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=187, end=193),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),
    ],
    # ===========================================================
    # ========================= INDEX 14 ========================
    # ===========================================================
    # Low-calorie artificial sweeteners like saccharin and acesulfame-K are known for their bitter aftertastes.
    # Recently, GIV3727 (4-(2,2,3-trimethylcyclopentyl) butanoic acid), a chemical that blocks saccharin and
    # acesulfame-K activation of multiple bitter taste receptors has been developed. In the study, the addition of the
    # bitter taste receptor antagonist GIV3727 to the saccharin and acesulfame-K solutions resulted in significantly
    # lower taste intensity ratings when compared to the solutions that were not treated with GIV3727. This suggests
    # that GIV3727 inhibits the normal functions of the bitter taste receptors because saccharin and acesulfame-K's
    # bitter aftertastes were not observed. The ability to inhibit activation of the bitter taste receptors can have
    # far-reaching effects if the bitter aftertastes of not only these two artificial sweeteners but also other foods,
    # beverages, and even pharmaceuticals can be minimized.
    [
        Annotation(anchor_text='saccharin', anchor_text_location=AnchorTextLocation(start=39, end=48),
                   entity_id='enwiki:Saccharin', entity_name='Saccharin'),

        Annotation(anchor_text='acesulfame-K', anchor_text_location=AnchorTextLocation(start=53, end=65),
                   entity_id='enwiki:Acesulfame%20potassium', entity_name='Acesulfame potassium'),

        Annotation(anchor_text='saccharin', anchor_text_location=AnchorTextLocation(start=195, end=204),
                   entity_id='enwiki:Saccharin', entity_name='Saccharin'),

        Annotation(anchor_text='acesulfame-K', anchor_text_location=AnchorTextLocation(start=209, end=221),
                   entity_id='enwiki:Acesulfame%20potassium', entity_name='Acesulfame potassium'),

        Annotation(anchor_text='taste receptor', anchor_text_location=AnchorTextLocation(start=252, end=266),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='taste receptor', anchor_text_location=AnchorTextLocation(start=329, end=343),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='receptor antagonist', anchor_text_location=AnchorTextLocation(start=335, end=354),
                   entity_id='enwiki:Receptor%20antagonist', entity_name='Receptor antagonist'),

        Annotation(anchor_text='saccharin', anchor_text_location=AnchorTextLocation(start=370, end=379),
                   entity_id='enwiki:Saccharin', entity_name='Saccharin'),

        Annotation(anchor_text='acesulfame-K', anchor_text_location=AnchorTextLocation(start=384, end=396),
                   entity_id='enwiki:Acesulfame%20potassium', entity_name='Acesulfame potassium'),

        Annotation(anchor_text='taste', anchor_text_location=AnchorTextLocation(start=439, end=444),
                   entity_id='enwiki:Taste', entity_name='Taste'),

        Annotation(anchor_text='bitter taste receptors', anchor_text_location=AnchorTextLocation(start=594, end=616),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='saccharin', anchor_text_location=AnchorTextLocation(start=625, end=634),
                   entity_id='enwiki:Saccharin', entity_name='Saccharin'),

        Annotation(anchor_text='acesulfame-K\'s', anchor_text_location=AnchorTextLocation(start=639, end=653),
                   entity_id='enwiki:Acesulfame%20potassium', entity_name='Acesulfame potassium'),

        Annotation(anchor_text='bitter taste receptors', anchor_text_location=AnchorTextLocation(start=733, end=755),
                   entity_id='enwiki:Taste%20receptor', entity_name='Taste receptor'),

        Annotation(anchor_text='foods', anchor_text_location=AnchorTextLocation(start=871, end=876),
                   entity_id='enwiki:Food', entity_name='Food'),

        Annotation(anchor_text='beverages', anchor_text_location=AnchorTextLocation(start=878, end=887),
                   entity_id='enwiki:Drink', entity_name='Drink'),
    ],
    # ===========================================================
    # ========================= INDEX 15 ========================
    # ===========================================================
    # In wine tasting the aftertaste or finish of a wine, is an important part of the evaluation. After tasting a wine,
    # a taster will determine the wine's aftertaste, which is a major determinant of the wine's quality. The aftertaste
    # of a wine can be described as bitter, persistent, short, sweet, smooth, or even non-existent. Included in
    # assessing the aftertaste of a wine is consideration of the aromas still present after swallowing. High quality
    # wines typically have long finishes accompanied by pleasant aromas. By assessing the combination of olfactory and
    # aftertaste sensations, wine tasting actually determines not only the aftertaste profile of a wine, but its flavor
    # profile as well.
    [
        Annotation(anchor_text='wine tasting', anchor_text_location=AnchorTextLocation(start=3, end=15),
                   entity_id='enwiki:Wine%20tasting', entity_name='Wine tasting'),

        Annotation(anchor_text='tasting a wine', anchor_text_location=AnchorTextLocation(start=98, end=112),
                   entity_id='enwiki:Wine%20tasting', entity_name='Wine tasting'),

        Annotation(anchor_text='taster', anchor_text_location=AnchorTextLocation(start=116, end=122),
                   entity_id='enwiki:Supertaster', entity_name='Supertaster'),

        Annotation(anchor_text='wine tasting', anchor_text_location=AnchorTextLocation(start=581, end=593),
                   entity_id='enwiki:Wine%20tasting', entity_name='Wine tasting'),

        Annotation(anchor_text='flavor', anchor_text_location=AnchorTextLocation(start=665, end=671),
                   entity_id='enwiki:Flavor', entity_name='Flavor'),
    ]

]

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#'enwiki:Blue-ringed%20octopus' annotations
annotation_dict['enwiki:Blue-ringed%20octopus'] = [
    # ===========================================================
    # ========================= INDEX 0 =========================
    # ===========================================================
    # The genus was described by British zoologist Guy Coburn Robson in 1929. There are four confirmed species of
    # Hapalochlaena, and six possible species still being researched:
    [
        Annotation(anchor_text='Guy Coburn Robson', anchor_text_location=AnchorTextLocation(start=45, end=62),
                   entity_id='enwiki:Guy%20Coburn%20Robson', entity_name='Guy Coburn Robson')
    ],
    # ===========================================================
    # ========================= INDEX 1 =========================
    # ===========================================================
    # Greater blue-ringed octopus (Hapalochlaena lunulata)
    [
        Annotation(anchor_text='Greater blue-ringed octopus', anchor_text_location=AnchorTextLocation(start=0, end=27),
                   entity_id='enwiki:Greater%20blue-ringed%20octopus', entity_name='Greater blue-ringed octopus')
    ],
    # ===========================================================
    # ========================= INDEX 2 =========================
    # ===========================================================
    # Southern blue-ringed octopus or lesser blue-ringed octopus (Hapalochlaena maculosa)
    [
        Annotation(anchor_text='Southern blue-ringed octopus', anchor_text_location=AnchorTextLocation(start=0, end=28),
                   entity_id='enwiki:Southern%20blue-ringed%20octopus', entity_name='Southern blue-ringed octopus')
    ],
    # ===========================================================
    # ========================= INDEX 3 =========================
    # ===========================================================
    # Blue-lined octopus (Hapalochlaena fasciata)
    [
        Annotation(anchor_text='Blue-lined octopus', anchor_text_location=AnchorTextLocation(start=0, end=18),
                   entity_id='enwiki:Blue-lined%20octopus', entity_name='Blue-lined octopus')
    ],
    # ===========================================================
    # ========================= INDEX 4 =========================
    # ===========================================================
    # Hapalochlaena nierstraszi was described in 1938 from a single specimen from the Bay of Bengal, with a
    # second specimen caught and described in 2013.
    [
        Annotation(anchor_text='Bay of Bengal', anchor_text_location=AnchorTextLocation(start=80, end=93),
                   entity_id='enwiki:Bay%20of%20Bengal', entity_name='Bay of Bengal')
    ],
    # ===========================================================
    # ========================= INDEX 5 =========================
    # ===========================================================
    # Blue-ringed octopuses spend much of their time hiding in crevices while displaying effective camouflage patterns
    # with their dermal chromatophore cells.  Like all octopuses, they can change shape easily, which helps them to
    # squeeze into crevices much smaller than themselves. This, along with piling up rocks outside the entrance to its
    # lair, helps safeguard the octopus from predators.
    [
        Annotation(anchor_text='chromatophore', anchor_text_location=AnchorTextLocation(start=131, end=144),
                   entity_id='enwiki:Chromatophore', entity_name='Chromatophore'),

        Annotation(anchor_text='octopuses', anchor_text_location=AnchorTextLocation(start=162, end=171),
                   entity_id='enwiki:Octopus', entity_name='Octopus'),

        Annotation(anchor_text='octopus', anchor_text_location=AnchorTextLocation(start=362, end=369),
                   entity_id='enwiki:Octopus', entity_name='Octopus')
    ],
    # ===========================================================
    # ========================= INDEX 6 =========================
    # ===========================================================
    # If they are provoked, they quickly change color, becoming bright yellow with each of the 50-60 rings flashing
    # bright iridescent blue within a third of a second as an aposematic warning display.  In the greater blue-ringed
    # octopus (Hapalochlaena lunulata), the rings contain multi-layer light reflectors called iridophores.  These are
    # arranged to reflect blue–green light in a wide viewing direction. Beneath and around each ring there are dark
    # pigmented chromatophores which can be expanded within 1 second to enhance the contrast of the rings. There are
    # no chromatophores above the ring, which is unusual for cephalopods as they typically use chromatophores to cover
    # or spectrally modify iridescence. The fast flashes of the blue rings are achieved using muscles which are under
    # neural control. Under normal circumstances, each ring is hidden by contraction of muscles above the iridophores.
    # When these relax and muscles outside the ring contract, the iridescence is exposed thereby revealing the blue
    # color.
    [
        Annotation(anchor_text='iridescent', anchor_text_location=AnchorTextLocation(start=117, end=127),
                   entity_id='enwiki:Iridescence', entity_name='Iridescence'),

        Annotation(anchor_text='aposematic', anchor_text_location=AnchorTextLocation(start=166, end=176),
                   entity_id='enwiki:Aposematism', entity_name='Aposematism'),

        Annotation(anchor_text='greater blue-ringed octopus', anchor_text_location=AnchorTextLocation(start=202, end=229),
                   entity_id='enwiki:Greater%20blue-ringed%20octopus', entity_name='Greater blue-ringed octopus'),

        Annotation(anchor_text='light reflectors', anchor_text_location=AnchorTextLocation(start=286, end=302),
                   entity_id='enwiki:Structural%20coloration', entity_name='Structural coloration'),

        Annotation(anchor_text='iridophores', anchor_text_location=AnchorTextLocation(start=310, end=321),
                   entity_id='enwiki:Chromatophore', entity_name='Chromatophore'),

        Annotation(anchor_text='dark pigmented chromatophores', anchor_text_location=AnchorTextLocation(start=439, end=468),
                   entity_id='enwiki:Chromatophore', entity_name='Chromatophore'),

        Annotation(anchor_text='chromatophores', anchor_text_location=AnchorTextLocation(start=558, end=572),
                   entity_id='enwiki:Chromatophore', entity_name='Chromatophore'),

        Annotation(anchor_text='chromatophores', anchor_text_location=AnchorTextLocation(start=644, end=658),
                   entity_id='enwiki:Chromatophore', entity_name='Chromatophore'),

        Annotation(anchor_text='iridescence', anchor_text_location=AnchorTextLocation(start=689, end=700),
                   entity_id='enwiki:Iridescence', entity_name='Iridescence'),

        Annotation(anchor_text='iridophores', anchor_text_location=AnchorTextLocation(start=880, end=891),
                   entity_id='enwiki:Chromatophore', entity_name='Chromatophore'),

        Annotation(anchor_text='iridescence', anchor_text_location=AnchorTextLocation(start=954, end=965),
                   entity_id='enwiki:Iridescence', entity_name='Iridescence'),

    ],
    # ===========================================================
    # ========================= INDEX 7 =========================
    # ===========================================================
    # In common with other Octopoda, the blue-ringed octopus swims by expelling water from a funnel in a form of jet
    # propulsion.
    [
        Annotation(anchor_text='Octopoda', anchor_text_location=AnchorTextLocation(start=21, end=29),
                   entity_id='enwiki:Octopus', entity_name='Octopus'),

        Annotation(anchor_text='funnel', anchor_text_location=AnchorTextLocation(start=87, end=93),
                   entity_id='enwiki:Siphon%20(mollusc)', entity_name='Siphon (mollusc)'),

        Annotation(anchor_text='jet propulsion', anchor_text_location=AnchorTextLocation(start=107, end=121),
                   entity_id='enwiki:Jet%20propulsion', entity_name='Jet propulsion')

    ],
    # ===========================================================
    # ========================= INDEX 8 =========================
    # ===========================================================
    # In common with other Octopoda, the blue-ringed octopus swims by expelling water from a funnel in a form of jet
    # propulsion.
    [
        Annotation(anchor_text='crab', anchor_text_location=AnchorTextLocation(start=57, end=61),
                   entity_id='enwiki:Crab', entity_name='Crab'),

        Annotation(anchor_text='shrimp', anchor_text_location=AnchorTextLocation(start=68, end=74),
                   entity_id='enwiki:Shrimp', entity_name='Shrimp'),

        Annotation(anchor_text='fish', anchor_text_location=AnchorTextLocation(start=102, end=106),
                   entity_id='enwiki:Fish', entity_name='Fish'),

        Annotation(anchor_text='horny beak', anchor_text_location=AnchorTextLocation(start=252, end=262),
                   entity_id='enwiki:Cephalopod%20beak', entity_name='Cephalopod beak'),

        Annotation(anchor_text='crab', anchor_text_location=AnchorTextLocation(start=291, end=295),
                   entity_id='enwiki:Crab', entity_name='Crab'),

        Annotation(anchor_text='shrimp', anchor_text_location=AnchorTextLocation(start=299, end=305),
                   entity_id='enwiki:Shrimp', entity_name='Shrimp'),

        Annotation(anchor_text='exoskeleton', anchor_text_location=AnchorTextLocation(start=306, end=317),
                   entity_id='enwiki:Exoskeleton', entity_name='Exoskeleton'),

        Annotation(anchor_text='paralyses', anchor_text_location=AnchorTextLocation(start=350, end=359),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),
    ],
    # ===========================================================
    # ========================= INDEX 9 =========================
    # ===========================================================
    # The mating ritual for the blue-ringed octopus begins when a male approaches a female and begins to caress her
    # with his modified arm, the hectocotylus. A male mates with a female by grabbing her, which sometimes completely
    # obscures the female's vision, then transferring sperm packets by inserting his hectocotylus into her mantle cavity
    # repeatedly. Mating continues until the female has had enough, and in at least one species the female has to remove
    # the over-enthusiastic male by force. Males will attempt copulation with members of their own species regardless
    # of sex or size, but interactions between males are most often shorter in duration and end with the mounting
    # octopus withdrawing the hectocotylus without packet insertion or struggle.
    [
        Annotation(anchor_text='hectocotylus', anchor_text_location=AnchorTextLocation(start=137, end=149),
                   entity_id='enwiki:Hectocotylus', entity_name='Hectocotylus'),

        Annotation(anchor_text='sperm', anchor_text_location=AnchorTextLocation(start=270, end=275),
                   entity_id='enwiki:Sperm', entity_name='Sperm'),

        Annotation(anchor_text='hectocotylus', anchor_text_location=AnchorTextLocation(start=301, end=313),
                   entity_id='enwiki:Hectocotylus', entity_name='Hectocotylus'),

        Annotation(anchor_text='hectocotylus', anchor_text_location=AnchorTextLocation(start=696, end=708),
                   entity_id='enwiki:Hectocotylus', entity_name='Hectocotylus'),

    ],
    # ===========================================================
    # ========================= INDEX 10 ========================
    # ===========================================================
    # Blue-ringed octopus females lay only one clutch of about 50 eggs in their lifetimes towards the end of autumn.
    # Eggs are laid then incubated underneath the female's arms for about six months, and during this process she
    # does not eat. After the eggs hatch, the female dies, and the new offspring will reach maturity and be able to
    # mate by the next year.
    [],
    # ===========================================================
    # ========================= INDEX 11 ========================
    # ===========================================================
    # The blue-ringed octopus, despite its small size, carries enough venom to kill twenty-six adult humans within
    # minutes. Their bites are tiny and often painless, with many victims not realizing they have been envenomated
    # until respiratory depression and paralysis start to set in. No blue-ringed octopus antivenom is available.
    [
        Annotation(anchor_text='envenomated', anchor_text_location=AnchorTextLocation(start=206, end=217),
                   entity_id='enwiki:Envenomation', entity_name='Envenomation'),

        Annotation(anchor_text='respiratory depression', anchor_text_location=AnchorTextLocation(start=224, end=246),
                   entity_id='enwiki:Hypoventilation', entity_name='Hypoventilation'),

        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=251, end=260),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),

        Annotation(anchor_text='antivenom', anchor_text_location=AnchorTextLocation(start=301, end=310),
                   entity_id='enwiki:Antivenom', entity_name='Antivenom'),
    ],
    # ===========================================================
    # ========================= INDEX 12 ========================
    # ===========================================================
    # The octopus produces venom containing tetrodotoxin, histamine, tryptamine, octopamine, taurine, acetylcholine and
    # dopamine. The venom can result in nausea, respiratory arrest, heart failure, severe and sometimes total paralysis,
    # blindness, and can lead to death within minutes if not treated. Death, if it occurs, is usually from suffocation
    # due to paralysis of the diaphragm.
    [
        Annotation(anchor_text='tetrodotoxin', anchor_text_location=AnchorTextLocation(start=38, end=50),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),

        Annotation(anchor_text='histamine', anchor_text_location=AnchorTextLocation(start=52, end=61),
                   entity_id='enwiki:Histamine', entity_name='Histamine'),

        Annotation(anchor_text='tryptamine', anchor_text_location=AnchorTextLocation(start=63, end=73),
                   entity_id='enwiki:Tryptamine', entity_name='Tryptamine'),

        Annotation(anchor_text='octopamine', anchor_text_location=AnchorTextLocation(start=75, end=85),
                   entity_id='enwiki:Octopamine%20(drug)', entity_name='Octopamine (drug)'),

        Annotation(anchor_text='taurine', anchor_text_location=AnchorTextLocation(start=87, end=94),
                   entity_id='enwiki:Taurine', entity_name='Taurine'),

        Annotation(anchor_text='acetylcholine', anchor_text_location=AnchorTextLocation(start=96, end=109),
                   entity_id='enwiki:Acetylcholine', entity_name='Acetylcholine'),

        Annotation(anchor_text='dopamine', anchor_text_location=AnchorTextLocation(start=114, end=122),
                   entity_id='enwiki:Dopamine', entity_name='Dopamine'),

        Annotation(anchor_text='nausea', anchor_text_location=AnchorTextLocation(start=148, end=154),
                   entity_id='enwiki:Nausea', entity_name='Nausea'),

        Annotation(anchor_text='respiratory arrest', anchor_text_location=AnchorTextLocation(start=156, end=174),
                   entity_id='enwiki:Respiratory%20arrest', entity_name='Respiratory arrest'),

        Annotation(anchor_text='heart failure', anchor_text_location=AnchorTextLocation(start=176, end=189),
                   entity_id='enwiki:Heart%20failure', entity_name='Heart failure'),

        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=218, end=227),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),

        Annotation(anchor_text='blindness', anchor_text_location=AnchorTextLocation(start=229, end=238),
                   entity_id='enwiki:Visual%20impairment', entity_name='Visual impairment'),

        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=349, end=358),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),
    ],
    # ===========================================================
    # ========================= INDEX 13 ========================
    # ===========================================================
    # The major neurotoxin component of the blue-ringed octopus is a compound that was originally known as maculotoxin
    # but was later found to be identical to tetrodotoxin, a neurotoxin also found in pufferfish, and in some poison
    # dart frogs. Tetrodotoxin is 1,200 times more toxic than cyanide. Tetrodotoxin blocks sodium channels, causing
    # motor paralysis, and respiratory arrest within minutes of exposure. The tetrodotoxin is produced by bacteria in
    # the salivary glands of the octopus.
    [
        Annotation(anchor_text='neurotoxin', anchor_text_location=AnchorTextLocation(start=10, end=20),
                   entity_id='enwiki:Neurotoxin', entity_name='Neurotoxin'),

        Annotation(anchor_text='maculotoxin', anchor_text_location=AnchorTextLocation(start=101, end=112),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),

        Annotation(anchor_text='tetrodotoxin', anchor_text_location=AnchorTextLocation(start=152, end=164),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),

        Annotation(anchor_text='neurotoxin', anchor_text_location=AnchorTextLocation(start=168, end=178),
                   entity_id='enwiki:Neurotoxin', entity_name='Neurotoxin'),

        Annotation(anchor_text='pufferfish', anchor_text_location=AnchorTextLocation(start=193, end=203),
                   entity_id='enwiki:Tetraodontidae', entity_name='Tetraodontidae'),

        Annotation(anchor_text='poison dart frog', anchor_text_location=AnchorTextLocation(start=217, end=233),
                   entity_id='enwiki:Poison%20dart%20frog', entity_name='Poison dart frog'),

        Annotation(anchor_text='Tetrodotoxin', anchor_text_location=AnchorTextLocation(start=236, end=248),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),

        Annotation(anchor_text='cyanide', anchor_text_location=AnchorTextLocation(start=280, end=287),
                   entity_id='enwiki:Cyanide', entity_name='Cyanide'),

        Annotation(anchor_text='Tetrodotoxin', anchor_text_location=AnchorTextLocation(start=289, end=301),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),

        Annotation(anchor_text='sodium channels', anchor_text_location=AnchorTextLocation(start=309, end=324),
                   entity_id='enwiki:Sodium%20channel', entity_name='Sodium channel'),

        Annotation(anchor_text='motor', anchor_text_location=AnchorTextLocation(start=334, end=339),
                   entity_id='enwiki:Motor%20system', entity_name='Motor system'),

        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=340, end=349),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),

        Annotation(anchor_text='respiratory arrest', anchor_text_location=AnchorTextLocation(start=355, end=373),
                   entity_id='enwiki:Respiratory%20arrest', entity_name='Respiratory arrest'),

        Annotation(anchor_text='tetrodotoxin', anchor_text_location=AnchorTextLocation(start=406, end=418),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),

        Annotation(anchor_text='bacteria', anchor_text_location=AnchorTextLocation(start=434, end=442),
                   entity_id='enwiki:Bacteria', entity_name='Bacteria'),
    ],
    # ===========================================================
    # ========================= INDEX 14 ========================
    # ===========================================================
    # A person must be in contact with the octopus to be envenomated. Faced with danger, the octopus's first instinct is
    # to flee. If the threat persists, the octopus will go into a defensive stance, and show its blue rings. Only if an
    # octopus is cornered, and touched, will a person be in danger of being bitten and envenomated.
    [
        Annotation(anchor_text='envenomated', anchor_text_location=AnchorTextLocation(start=51, end=62),
                   entity_id='enwiki:Envenomation', entity_name='Envenomation'),

        Annotation(anchor_text='envenomated', anchor_text_location=AnchorTextLocation(start=310, end=321),
                   entity_id='enwiki:Envenomation', entity_name='Envenomation'),
    ],
    # ===========================================================
    # ========================= INDEX 15 ========================
    # ===========================================================
    # Tetrodotoxin causes severe and often total body paralysis. Tetrodotoxin envenomation can result in victims being
    # fully aware of their surroundings but unable to breathe. Because of the paralysis that occurs, they have no way
    # of signaling for help or any way of indicating distress. The victim remains conscious and alert in a manner
    # similar to curare or pancuronium bromide. This effect, however, is temporary and will fade over a period of hours
    # as the tetrodotoxin is metabolized and excreted by the body.
    [
        Annotation(anchor_text='Tetrodotoxin', anchor_text_location=AnchorTextLocation(start=0, end=12),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),

        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=48, end=57),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),

        Annotation(anchor_text='Tetrodotoxin', anchor_text_location=AnchorTextLocation(start=59, end=71),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),

        Annotation(anchor_text='envenomation', anchor_text_location=AnchorTextLocation(start=72, end=84),
                   entity_id='enwiki:Envenomation', entity_name='Envenomation'),

        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=185, end=194),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),

        Annotation(anchor_text='curare', anchor_text_location=AnchorTextLocation(start=344, end=350),
                   entity_id='enwiki:Curare', entity_name='Curare'),

        Annotation(anchor_text='pancuronium bromide', anchor_text_location=AnchorTextLocation(start=354, end=373),
                   entity_id='enwiki:Pancuronium%20bromide', entity_name='Pancuronium bromide'),

        Annotation(anchor_text='tetrodotoxin', anchor_text_location=AnchorTextLocation(start=454, end=466),
                   entity_id='enwiki:Tetrodotoxin', entity_name='Tetrodotoxin'),
    ],
    # ===========================================================
    # ========================= INDEX 16 ========================
    # ===========================================================
    # The symptoms vary in severity, with children being the most at risk because of their small body size.
    [],
    # ===========================================================
    # ========================= INDEX 17 ========================
    # ===========================================================
    # First aid treatment is pressure on the wound and artificial respiration once the paralysis has disabled the
    # victim's respiratory muscles, which often occurs within minutes of being bitten. Because the venom primarily kills
    # through paralysis, victims are frequently saved if artificial respiration is started and maintained before marked
    # cyanosis and hypotension develop. Efforts should be continued even if the victim appears not to be responding.
    # Respiratory support until medical assistance arrives ensures the victims will generally recover.
    [
        Annotation(anchor_text='First aid', anchor_text_location=AnchorTextLocation(start=0, end=9),
                   entity_id='enwiki:First%20aid', entity_name='First aid'),

        Annotation(anchor_text='artificial respiration', anchor_text_location=AnchorTextLocation(start=49, end=71),
                   entity_id='enwiki:Artificial%20ventilation', entity_name='Artificial ventilation'),

        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=81, end=90),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),

        Annotation(anchor_text='disabled the victim\'s respiratory muscles', anchor_text_location=AnchorTextLocation(start=95, end=136),
                   entity_id='enwiki:Respiratory%20arrest', entity_name='Respiratory arrest'),

        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=231, end=240),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),

        Annotation(anchor_text='artificial respiration', anchor_text_location=AnchorTextLocation(start=274, end=296),
                   entity_id='enwiki:Artificial%20ventilation', entity_name='Artificial ventilation'),

        Annotation(anchor_text='cyanosis', anchor_text_location=AnchorTextLocation(start=337, end=345),
                   entity_id='enwiki:Cyanosis', entity_name='Cyanosis'),

        Annotation(anchor_text='hypotension', anchor_text_location=AnchorTextLocation(start=350, end=361),
                   entity_id='enwiki:Hypotension', entity_name='Hypotension'),
    ],
    # ===========================================================
    # ========================= INDEX 18 ========================
    # ===========================================================
    # It is essential that rescue breathing be continued without pause until the paralysis subsides and the victim
    # regains the ability to breathe on their own. This is a daunting physical prospect for a single individual, but
    # use of a bag valve mask respirator reduces fatigue to sustainable levels until help can arrive.
    [
        Annotation(anchor_text='paralysis', anchor_text_location=AnchorTextLocation(start=75, end=84),
                   entity_id='enwiki:Paralysis', entity_name='Paralysis'),

        Annotation(anchor_text='bag valve mask', anchor_text_location=AnchorTextLocation(start=229, end=243),
                   entity_id='enwiki:Bag%20valve%20mask', entity_name='Bag valve mask'),

    ],
    # ===========================================================
    # ========================= INDEX 19 ========================
    # ===========================================================
    # Definitive hospital treatment involves placing the patient on a medical ventilator until the toxin is removed by
    # the body.
    [
        Annotation(anchor_text='hospital', anchor_text_location=AnchorTextLocation(start=11, end=19),
                   entity_id='enwiki:Hospital', entity_name='Hospital'),

        Annotation(anchor_text='medical ventilator', anchor_text_location=AnchorTextLocation(start=64, end=82),
                   entity_id='enwiki:Medical%20ventilator', entity_name='Medical ventilator'),
    ],
    # ===========================================================
    # ========================= INDEX 20 ========================
    # ===========================================================
    # Victims who survive the first twenty-four hours usually recover completely.
    [],
    # ===========================================================
    # ========================= INDEX 21 ========================
    # ===========================================================
    # The blue-ringed octopus is the prominent symbol of the secret order of female bandits and smugglers in the James
    # Bond film Octopussy, appearing in an aquarium tank, on silk robes, and as a tattoo on women in the order.  The
    # animal was also featured in the book State of Fear by Michael Crichton, where a terrorist organization utilized
    # the animal's venom as a favored murder weapon. The Adventure Zone featured a blue-ringed octopus in its "Petals
    # To The Metal" series.
    [
        Annotation(anchor_text='James Bond', anchor_text_location=AnchorTextLocation(start=107, end=117),
                   entity_id='enwiki:James%20Bond', entity_name='James Bond'),

        Annotation(anchor_text='Octopussy', anchor_text_location=AnchorTextLocation(start=123, end=132),
                   entity_id='enwiki:Octopussy', entity_name='Octopussy'),

        Annotation(anchor_text='State of Fear', anchor_text_location=AnchorTextLocation(start=261, end=274),
                   entity_id='enwiki:State%20of%20Fear', entity_name='State of Fear'),

        Annotation(anchor_text='Michael Crichton', anchor_text_location=AnchorTextLocation(start=278, end=294),
                   entity_id='enwiki:Michael%20Crichton', entity_name='Michael Crichton'),

        Annotation(anchor_text='The Adventure Zone', anchor_text_location=AnchorTextLocation(start=383, end=401),
                   entity_id='enwiki:The%20Adventure%20Zone', entity_name='The Adventure Zone'),
    ],

]

# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# 'enwiki:Aerobic%20fermentation' annotations
annotation_dict['enwiki:Aerobic%20fermentation'] = [
    # ===========================================================
    # ========================= INDEX 0 =========================
    # ===========================================================
    # Approximately 100 million years ago (mya), within the yeast lineage there was a whole genome duplication (WGD).
    # A majority of Crabtree-positive yeasts are post-WGD yeasts.  It was believed that the WGD was a mechanism for the
    # development of Crabtree effect in these species due to the duplication of alcohol dehydrogenase (ADH) encoding
    # genes and hexose transporters.  However, recent evidence has shown that aerobic fermentation originated before
    # the WGD and evolved as a multi-step process, potentially aided by the WGD.  The origin of aerobic fermentation,
    # or the first step, in Saccharomyces crabtree-positive yeasts likely occurred in the interval between the ability
    # to grow under anaerobic conditions, horizontal transfer of anaerobic DHODase (encoded by URA1 with bacteria), and
    # the loss of respiratory chain Complex I.  A more pronounced Crabtree effect, the second step, likely occurred
    # near the time of the WGD event.  Later evolutionary events that aided in the evolution of aerobic fermentation
    # are better understood and outlined in the Genomic basis of the crabtree effect section.
    [
        Annotation(anchor_text='whole genome duplication', anchor_text_location=AnchorTextLocation(start=80, end=104),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='post-WGD', anchor_text_location=AnchorTextLocation(start=156, end=164),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='WGD', anchor_text_location=AnchorTextLocation(start=199, end=202),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='alcohol dehydrogenase', anchor_text_location=AnchorTextLocation(start=301, end=322),
                   entity_id='enwiki:Alcohol%20dehydrogenase', entity_name='Alcohol dehydrogenase'),

        Annotation(anchor_text='WGD', anchor_text_location=AnchorTextLocation(start=453, end=456),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='WGD', anchor_text_location=AnchorTextLocation(start=519, end=522),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='WGD event', anchor_text_location=AnchorTextLocation(start=919, end=928),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

    ],
    # ===========================================================
    # ========================= INDEX 1 =========================
    # ===========================================================
    # It is believed that a major driving force in the origin of aerobic fermentation was its simultaneous origin with
    # modern fruit (~125 mya).  These fruit provided an abundance of simple sugar food source for microbial communities,
    # including both yeast and bacteria.  Bacteria, at that time, were able to produce biomass at a faster rate than the
    # yeast.  Producing a toxic compound, like ethanol, can slow the growth of bacteria, allowing the yeast to be more
    # competitive.  However, the yeast still had to use a portion of the sugar it consumes to produce ethanol.
    # Crabtree-positive yeasts also have increased glycolytic flow, or increased uptake of glucose and conversion to
    # pyruvate, which compensates for using a portion of the glucose to produce ethanol rather than biomass.  Therefore,
    # it is believed that the original driving force was to kill competitors.  This is supported by research that
    # determined the kinetic behavior of the ancestral ADH protein, which was found to be optimized to make ethanol,
    # rather than consume it.
    [],
    # ===========================================================
    # ========================= INDEX 2 =========================
    # ===========================================================
    # Further evolutionary events in the development of aerobic fermentation likely increased the efficiency of this
    # lifestyle, including increased tolerance to ethanol and the repression of the respiratory pathway.  In high sugar
    # environments, S. cerevisiae outcompetes and dominants all other yeast species, except its closest relative
    # Saccharomyces paradoxus. The ability of S. cerevisiae to dominate in high sugar environments evolved more
    # recently than aerobic fermentation and is dependent on the type of high-sugar environment.  Other yeasts' growth
    # is dependent on the pH and nutrients of the high-sugar environment.
    [
        Annotation(anchor_text='Saccharomyces paradoxus', anchor_text_location=AnchorTextLocation(start=332, end=355),
                   entity_id='enwiki:Saccharomyces%20paradoxus', entity_name='Saccharomyces paradoxus'),
    ],
    # ===========================================================
    # ========================= INDEX 3 =========================
    # ===========================================================
    # The genomic basis of the crabtree effect is still being invested, and its evolution likely involved multiple
    # successive molecular steps that increased the efficiency of the lifestyle.
    [],
    # ===========================================================
    # ========================= INDEX 4 =========================
    # ===========================================================
    # Hexose transporters (HXT) are a group of proteins that are largely responsible for the uptake of glucose in yeast.
    # In S. cerevisiae, 20 HXT genes have been identified and 17 encode for glucose transporters (HXT1-HXT17), GAL2
    # encodes for a galactose transporter, and SNF3 and RGT2 encode for glucose sensors.  The number of glucose sensor
    # genes have remained mostly consistent through the budding yeast lineage, however glucose sensors are absent from
    # Schizosaccharomyces pombe.  Sch. pombe is a Crabtree-positive yeast, which developed aerobic fermentation
    # independently from Saccharomyces lineage, and detects glucose via the cAMP-signaling pathway.  The number of
    # transporter genes vary significantly between yeast species and has continually increased during the evolution
    # of the S. cerevisiae lineage.  Most of the transporter genes have been generated by tandem duplication, rather
    # than from the WGD.  Sch. pombe also has a high number of transporter genes compared to its close relatives.
    # Glucose uptake is believed to be a major rate-limiting step in glycolysis and replacing S. cerevisiae
    [
        Annotation(anchor_text='Hexose transporters', anchor_text_location=AnchorTextLocation(start=0, end=19),
                   entity_id='enwiki:Glucose%20transporter', entity_name='Glucose transporter'),

        Annotation(anchor_text='HXT', anchor_text_location=AnchorTextLocation(start=137, end=140),
                   entity_id='enwiki:Glucose%20transporter', entity_name='Glucose transporter'),

        Annotation(anchor_text='glucose transporters', anchor_text_location=AnchorTextLocation(start=186, end=206),
                   entity_id='enwiki:Glucose%20transporter', entity_name='Glucose transporter'),

        Annotation(anchor_text='Schizosaccharomyces pombe', anchor_text_location=AnchorTextLocation(start=452, end=477),
                   entity_id='enwiki:Schizosaccharomyces%20pombe', entity_name='Schizosaccharomyces pombe'),

        Annotation(anchor_text='transporter genes', anchor_text_location=AnchorTextLocation(start=667, end=684),
                   entity_id='enwiki:Glucose%20transporter', entity_name='Glucose transporter'),

        Annotation(anchor_text='WGD', anchor_text_location=AnchorTextLocation(start=902, end=905),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='transporter genes', anchor_text_location=AnchorTextLocation(start=945, end=962),
                   entity_id='enwiki:Glucose%20transporter', entity_name='Glucose transporter'),

        Annotation(anchor_text='glycolysis', anchor_text_location=AnchorTextLocation(start=1060, end=1070),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),
    ],
    # ===========================================================
    # ========================= INDEX 5 =========================
    # ===========================================================
    # 's HXT1-17 genes with a single chimera HXT gene results in decreased ethanol production or fully respiratory
    # metabolism.  Thus, having an efficient glucose uptake system appears to be essential to ability of aerobic
    # fermentation.  There is a significant positive correlation between the number of hexose transporter genes and the
    # efficiency of ethanol production.
    [
        Annotation(anchor_text='HXT', anchor_text_location=AnchorTextLocation(start=39, end=42),
                   entity_id='enwiki:Glucose%20transporter', entity_name='Glucose transporter'),
    ],
    # ===========================================================
    # ========================= INDEX 6 =========================
    # ===========================================================
    # After a WGD, one of the duplicated gene pair is often lost through fractionation; less than 10% of WGD gene pairs
    # have remained in S. cerevisiae genome.  A little over half of WGD gene pairs in the glycolysis reaction pathway
    # were retained in post-WGD species, significantly higher than the overall retention rate.  This has been associated
    # with an increased ability to metabolize glucose into pyruvate, or higher rate of glycolysis.  After glycolysis,
    # pyruvate can either be further broken down by pyruvate decarboxylase (Pdc) or pyruvate dehydrogenase (Pdh).  The
    # kinetics of the enzymes are such that when pyruvate concentrations are high, due to a high rate of glycolysis,
    # there is increased flux through Pdc and thus the fermentation pathway.  The WGD is believed to have played a
    # beneficial role in the evolution of the Crabtree effect in post-WGD species partially due to this increase in
    # copy number of glycolysis genes.
    [
        Annotation(anchor_text='WGD', anchor_text_location=AnchorTextLocation(start=8, end=11),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='WGD', anchor_text_location=AnchorTextLocation(start=99, end=102),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='WGD', anchor_text_location=AnchorTextLocation(start=176, end=179),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='glycolysis', anchor_text_location=AnchorTextLocation(start=198, end=208),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

        Annotation(anchor_text='post-WGD', anchor_text_location=AnchorTextLocation(start=243, end=251),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='pyruvate decarboxylase', anchor_text_location=AnchorTextLocation(start=499, end=521),
                   entity_id='enwiki:Pyruvate%20decarboxylase', entity_name='Pyruvate decarboxylase'),

        Annotation(anchor_text='pyruvate dehydrogenase', anchor_text_location=AnchorTextLocation(start=531, end=553),
                   entity_id='enwiki:Pyruvate%20dehydrogenase', entity_name='Pyruvate dehydrogenase'),

        Annotation(anchor_text='glycolysis', anchor_text_location=AnchorTextLocation(start=665, end=675),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

        Annotation(anchor_text='Pdc', anchor_text_location=AnchorTextLocation(start=709, end=712),
                   entity_id='enwiki:Pyruvate%20decarboxylase', entity_name='Pyruvate decarboxylase'),

        Annotation(anchor_text='WGD', anchor_text_location=AnchorTextLocation(start=753, end=756),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='post-WGD', anchor_text_location=AnchorTextLocation(start=845, end=853),
                   entity_id='enwiki:Paleopolyploidy', entity_name='Paleopolyploidy'),

        Annotation(anchor_text='glycolysis genes', anchor_text_location=AnchorTextLocation(start=911, end=927),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

    ],
    # ===========================================================
    # ========================= INDEX 7 =========================
    # ===========================================================
    # The fermentation reaction only involves two steps.  Pyruvate is converted to acetaldehyde by Pdc and then
    # acetaldehyde is converted to ethanol by alcohol dehydrogenase (Adh).  There is no significant increase in the
    # number of Pdc genes in Crabtree-positive compared to Crabtree-negative species and no correlation between number
    # of Pdc genes and efficiency of fermentation.  There are five Adh genes in S. cerevisiae.  Adh1 is the major enzyme
    # responsible for catalyzing the fermentation step from acetaldehyde to ethanol.  Adh2 catalyzes the reverse
    # reaction, consuming ethanol and converting it to acetaldehyde.  The ancestral, or original, Adh had a similar
    # function as Adh1 and after a duplication in this gene, Adh2 evolved a lower K for ethanol.  Adh2 is believed to
    # have increased yeast species' tolerance for ethanol and allowed Crabtree-positive species to consume the ethanol
    # they produced after depleting sugars.  However, Adh2 and consumption of ethanol is not essential for aerobic
    # fermentation.  Sch. pombe and other Crabtree positive species do not have the ADH2  gene and consumes ethanol
    # very poorly.
    [
        Annotation(anchor_text='Pdc', anchor_text_location=AnchorTextLocation(start=93, end=96),
                   entity_id='enwiki:Pyruvate%20decarboxylase', entity_name='Pyruvate decarboxylase'),

        Annotation(anchor_text='alcohol dehydrogenase', anchor_text_location=AnchorTextLocation(start=146, end=167),
                   entity_id='enwiki:Alcohol%20dehydrogenase', entity_name='Alcohol dehydrogenase'),

        Annotation(anchor_text='Pdc genes', anchor_text_location=AnchorTextLocation(start=226, end=235),
                   entity_id='enwiki:Pyruvate%20decarboxylase', entity_name='Pyruvate decarboxylase'),

        Annotation(anchor_text='Pdc genes', anchor_text_location=AnchorTextLocation(start=332, end=341),
                   entity_id='enwiki:Pyruvate%20decarboxylase', entity_name='Pyruvate decarboxylase'),

        Annotation(anchor_text='Adh', anchor_text_location=AnchorTextLocation(start=643, end=646),
                   entity_id='enwiki:Alcohol%20dehydrogenase', entity_name='Alcohol dehydrogenase'),
    ],
    # ===========================================================
    # ========================= INDEX 8 =========================
    # ===========================================================
    # In Crabtree-negative species, respiration related genes are highly expressed in the presence of oxygen.  However,
    # when S. cerevisiae is grown on glucose in aerobic conditions, respiration-related gene expression is repressed.
    # Mitochondrial ribosomal proteins expression is only induced under environmental stress conditions, specifically
    # low glucose availability.  Genes involving mitochondrial energy generation and phosphorylation oxidation, which
    # are involved in respiration, have the largest expression difference between aerobic fermentative yeast species
    # and respiratory species.  In a comparative analysis between Sch. pombe and S. cerevisiae, both of which evolved
    # aerobic fermentation independently, the expression pattern of these two fermentative yeasts were more similar to
    # each other than a respiratory yeast, C. albicans.  However, S. cerevisiae is evolutionarily closer to C. albicans.
    # Regulatory rewiring was likely important in the evolution of aerobic fermentation in both lineages.
    [],
    # ===========================================================
    # ========================= INDEX 9 =========================
    # ===========================================================
    # Aerobic fermentation is also essential for multiple industries, resulting in human domestication of several yeast
    # strains.  Beer and other alcoholic beverages, throughout human history, have played a significant role in society
    # through drinking rituals, providing nutrition, medicine, and uncontaminated water.  During the domestication
    # process, organisms shift from natural environments that are more variable and complex to simple and stable
    # environments with a constant substrate. This often favors specialization adaptations in domesticated microbes,
    # associated with relaxed selection for non-useful genes in alternative metabolic strategies or pathogenicity.
    # Domestication might be partially responsible for the traits that promote aerobic fermentation in industrial
    # species.  Introgression and HGT is common in Saccharomyces domesticated strains.  Many commercial wine strains
    # have significant portions of their DNA derived from HGT of non-Saccharomyces species.  HGT and introgression are
    # less common in nature than is seen during domestication pressures.  For example, the important industrial yeast
    # strain Saccharomyces pastorianus, is an interspecies hybrid of S. cerevisiae and the cold tolerant S. eubayanus.
    # This hybrid is commonly used in lager-brewing, which requires slow, low temperature fermentation.
    [
        Annotation(anchor_text='Saccharomyces pastorianus', anchor_text_location=AnchorTextLocation(start=1116, end=1141),
                   entity_id='enwiki:Saccharomyces%20pastorianus', entity_name='Saccharomyces pastorianus'),

        Annotation(anchor_text='S. eubayanus', anchor_text_location=AnchorTextLocation(start=1208, end=1220),
                   entity_id='enwiki:Saccharomyces%20eubayanus', entity_name='Saccharomyces eubayanus'),
    ],
    # ===========================================================
    # ========================= INDEX 10 ========================
    # ===========================================================
    # Alcoholic fermentation is often used by plants in anaerobic conditions to produce ATP and regenerate NAD to allow
    # for glycolysis to continue.  For most plant tissues, fermentation only occurs in anaerobic conditions, but there
    # are a few exceptions.  In the pollen of maize (Zea mays) and tobacco (Nicotiana tabacum & Nicotiana
    # plumbaginifolia), the fermentation enzyme ADH is abundant, regardless of the oxygen level.  In tobacco pollen,
    # PDC is also highly expressed in this tissue and transcript levels are not influenced by oxygen concentration.
    # Tobacco pollen, similar to Crabtree-positive yeast, perform high levels of fermentation dependent on the sugar
    # supply, and not oxygen availability.  In these tissues, respiration and alcoholic fermentation occur
    # simultaneously with high sugar availability.  Fermentation produces the toxic acetaldehyde and ethanol, that can
    # build up in large quantities during pollen development.  It has been hypothesized that acetaldehyde is a pollen
    # factor that causes cytoplasmic male sterility.  Cytoplasmic male sterility is a trait observed in maize, tobacco
    # and other plants in which there is an inability to produce viable pollen.  It is believed that this trait might
    # be due to the expression of the fermentation genes, ADH and PDC, a lot earlier on in pollen development than
    # normal and the accumulation of toxic aldehyde.
    [
        Annotation(anchor_text='ATP', anchor_text_location=AnchorTextLocation(start=82, end=85),
                   entity_id='enwiki:Adenosine%20triphosphate', entity_name='Adenosine triphosphate'),

        Annotation(anchor_text='NAD', anchor_text_location=AnchorTextLocation(start=101, end=104),
                   entity_id='enwiki:Nicotinamide%20adenine%20dinucleotide', entity_name='Nicotinamide adenine dinucleotide'),

        Annotation(anchor_text='glycolysis', anchor_text_location=AnchorTextLocation(start=118, end=128),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

        Annotation(anchor_text='maize', anchor_text_location=AnchorTextLocation(start=267, end=272),
                   entity_id='enwiki:Maize', entity_name='Maize'),

        Annotation(anchor_text='tobacco', anchor_text_location=AnchorTextLocation(start=288, end=295),
                   entity_id='enwiki:Tobacco', entity_name='Tobacco'),

        Annotation(anchor_text='ADH', anchor_text_location=AnchorTextLocation(start=369, end=372),
                   entity_id='enwiki:Alcohol%20dehydrogenase', entity_name='Alcohol dehydrogenase'),

        Annotation(anchor_text='tobacco pollen', anchor_text_location=AnchorTextLocation(start=422, end=436),
                   entity_id='enwiki:Tobacco', entity_name='Tobacco'),

        Annotation(anchor_text='Tobacco pollen', anchor_text_location=AnchorTextLocation(start=549, end=563),
                   entity_id='enwiki:Tobacco', entity_name='Tobacco'),

        Annotation(anchor_text='cytoplasmic male sterility', anchor_text_location=AnchorTextLocation(start=1005, end=1031),
                   entity_id='enwiki:Cytoplasmic%20male%20sterility', entity_name='Cytoplasmic male sterility'),

        Annotation(anchor_text='Cytoplasmic male sterility', anchor_text_location=AnchorTextLocation(start=1034, end=1060),
                   entity_id='enwiki:Cytoplasmic%20male%20sterility', entity_name='Cytoplasmic male sterility'),

        Annotation(anchor_text='maize', anchor_text_location=AnchorTextLocation(start=1084, end=1089),
                   entity_id='enwiki:Maize', entity_name='Maize'),

        Annotation(anchor_text='tobacco', anchor_text_location=AnchorTextLocation(start=1091, end=1098),
                   entity_id='enwiki:Tobacco', entity_name='Tobacco'),

        Annotation(anchor_text='ADH', anchor_text_location=AnchorTextLocation(start=1263, end=1266),
                   entity_id='enwiki:Alcohol%20dehydrogenase', entity_name='Alcohol dehydrogenase'),

        Annotation(anchor_text='PDC', anchor_text_location=AnchorTextLocation(start=1271, end=1274),
                   entity_id='enwiki:Pyruvate%20decarboxylase', entity_name='Pyruvate decarboxylase'),
    ],
    # ===========================================================
    # ========================= INDEX 11 ========================
    # ===========================================================
    # When grown in glucose-rich media, trypanosomatid parasites degrade glucose via aerobic fermentation.  In this
    # group, this phenomenon is not a pre-adaptation to/or remnant of anaerobic life, shown through their inability to
    # survive in anaerobic conditions.  It is believed that this phenomenon developed due to the capacity for a high
    # glycolytic flux and the high glucose concentrations of their natural environment.  The mechanism for repression
    # of respiration in these conditions is not yet known.
    [
        Annotation(anchor_text='trypanosomatid', anchor_text_location=AnchorTextLocation(start=34, end=48),
                   entity_id='enwiki:Trypanosomatida', entity_name='Trypanosomatida'),

    ],
    # ===========================================================
    # ========================= INDEX 12 ========================
    # ===========================================================
    # A couple Escherichia coli mutant strains have been bioengineered to ferment glucose under aerobic conditions.
    # One group developed the ECOM3 (E. coli cytochrome oxidase mutant) strain by removing three terminal cytochrome
    # oxidases (cydAB, cyoABCD, and cbdAB) to reduce oxygen uptake.  After 60 days of adaptive evolution on glucose
    # media, the strain displayed a mixed phenotype.  In aerobic conditions, some populations' fermentation solely
    # produced lactate, while others did mixed-acid fermentation.
    [],
    # ===========================================================
    # ========================= INDEX 13 ========================
    # ===========================================================
    # One of the hallmarks of cancer is altered metabolism or deregulating cellular energetics.  Cancers cells often
    # have reprogrammed their glucose metabolism to perform lactic acid fermentation, in the presence of oxygen, rather
    # than send the pyruvate made through glycolysis to the mitochondria. This is referred to as the Warburg effect,
    # and is associated with high consumption of glucose and a high rate of glycolysis.   ATP production in these cancer
    # cells is often only through the process of glycolysis and pyruvate is broken down by the fermentation process in
    # the cell's cytoplasm.  This phenomenon is often seen as counterintuitive, since cancer cells have higher energy
    # demands due to the continued proliferation and respiration produces significantly more ATP than glycolysis alone
    # (fermentation produces no additional ATP).  Typically, there is an up-regulation in glucose transporters and
    # enzymes in the glycolysis pathway (also seen in yeast).  There are many parallel aspects of aerobic fermentation
    # in tumor cells that are also seen in Crabtree-positive yeasts. Further research into the evolution of aerobic
    # fermentation in yeast such as S. cerevisiae can be a useful model for understanding aerobic fermentation in tumor
    # cells.  This has a potential for better understanding cancer and cancer treatments.
    [
        Annotation(anchor_text='glycolysis', anchor_text_location=AnchorTextLocation(start=261, end=271),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

        Annotation(anchor_text='Warburg effect', anchor_text_location=AnchorTextLocation(start=320, end=334),
                   entity_id='enwiki:Warburg%20effect', entity_name='Warburg effect'),

        Annotation(anchor_text='glycolysis', anchor_text_location=AnchorTextLocation(start=406, end=416),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

        Annotation(anchor_text='ATP', anchor_text_location=AnchorTextLocation(start=420, end=423),
                   entity_id='enwiki:Adenosine%20triphosphate', entity_name='Adenosine triphosphate'),

        Annotation(anchor_text='glycolysis', anchor_text_location=AnchorTextLocation(start=494, end=504),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

        Annotation(anchor_text='ATP', anchor_text_location=AnchorTextLocation(start=763, end=766),
                   entity_id='enwiki:Adenosine%20triphosphate', entity_name='Adenosine triphosphate'),

        Annotation(anchor_text='glycolysis', anchor_text_location=AnchorTextLocation(start=772, end=782),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

        Annotation(anchor_text='ATP', anchor_text_location=AnchorTextLocation(start=826, end=829),
                   entity_id='enwiki:Adenosine%20triphosphate', entity_name='Adenosine triphosphate'),

        Annotation(anchor_text='glucose transporters', anchor_text_location=AnchorTextLocation(start=873, end=893),
                   entity_id='enwiki:Glucose%20transporter', entity_name='Glucose transporter'),

        Annotation(anchor_text='glycolysis pathway', anchor_text_location=AnchorTextLocation(start=913, end=931),
                   entity_id='enwiki:Glycolysis', entity_name='Glycolysis'),

    ],
]