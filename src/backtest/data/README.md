# Data namespace
This namespace contains useful tools to handle complex data pipelines.

## DataPipe
The DataPipe class, but also a way of writing pipelines.  From this class is derived an api to handle complex data 
pipelines that are written in an easy to read and maintain way.  It work by overriding the pipe operator `|` to chain
methods together.
Consider resetting the cache when changing the pipeline.  Failing to doing so might result in unexpected behaviour.
To reset the cache, simply delete the .cache directory.
### Under the hood
**get(from: datetime, to: datetime, \*args, \*\*kwargs)**  
A method calling the run method of every pipe in the pipeline following the predefined order.  It gives to the next pipe
the output of the previous one.  It calls the run method of each pipe in the pipeline.

**run(from: datetime, to: datetime, \*args, \*\*kwargs)**  
A method that shouldn't be overridden.  It calls the fetch, process, cache, revalidate, and collate methods if they are
registered.  It packages the data in a PipeOutput object and returns it.

**fetch(from: datetime, to: datetime, po: PipeOutput, \*args, \*\*kwargs)**  
A method that can be overridden.  It fetches the data from external sources.

**process(from: datetime, to: datetime, po: PipeOutput, \*args, \*\*kwargs)**  
A method that can be overridden.  It is designed to do static preprocessing on the data.

**cache(from: datetime, to: datetime, po: PipeOutput, \*args, \*\*kwargs)**  
A method that can be overridden.  It is designed to cache the data.  If the revalidate method upstream doesn't trigger a 
revalidate action or the revalidate time is not reached, the pipeline will use the cached data.
It must return the next revalidation time.

**revalidate(from: datetime, to: datetime, po: PipeOutput, \*args, \*\*kwargs)**  
A method that can be overridden.  Based on the po or the time, it can return a RevalidateAction.  This action can 
revalidate the whole pipeline or just
the downstream pipes.

**collate(from: datetime, to: datetime, po1: PipeOutput, po2: PipeOutput, \*args, \*\*kwargs)**  
A method that can be overridden.  This method aggregates the data from two different pipelines into one output.  
It is used to merge data from different sources.  This method will branch the pipe allowing the two branches to run
concurrently.

**_build(other: DataPipe)**  
Return a new DataPipe with a updated get method that merge the two pipes.

**\_\_or\_\_(other: DataPipe)**
Uses the build method to merge the two pipes.



## TODO
- [X] Continue JSONEncoder and JSONDecoder to make them fully functional (+ Tests)
- [X] Finish the JSONCache pipe
- [X] Add the FetchTickers pipe to the utils