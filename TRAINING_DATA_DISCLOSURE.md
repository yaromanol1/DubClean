# Training-data disclosure

DubClean model cards may describe training data in aggregate without publishing
the titles of individual films or recordings.

Only verified facts should be published. A useful public summary includes:

- the number of source works and the total usable audio duration;
- languages, translation types and broad genre distribution;
- channel layouts, sample rates and other relevant technical properties;
- how pairs, excerpts and train/validation/test splits were formed;
- preprocessing and filtering steps;
- known gaps, biases and limitations;
- the lawful basis under which the maintainer created and used the training
  material.

Do not invent missing counts or describe material as "open data" unless the
rights and source terms actually permit that statement.

Exact titles are not required by the DubClean license and do not need to appear
in the public repository or model card. The maintainer should nevertheless keep
a private provenance ledger containing, for each source:

- a stable internal identifier;
- origin and acquisition date;
- applicable license, permission or other lawful basis;
- permitted uses and redistribution limits;
- excerpts or derived files included in each dataset version;
- removal or dispute history.

That private ledger must not itself contain or expose personal information that
is unnecessary for rights management.

Publishing DubClean source code or weights does not grant rights to the films,
translations, recordings or other material used to create a dataset. Dataset
rights and model-distribution rights must be assessed separately for each
release.
