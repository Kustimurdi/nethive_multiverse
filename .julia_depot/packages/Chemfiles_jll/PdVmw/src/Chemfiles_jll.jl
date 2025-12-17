# Use baremodule to shave off a few KB from the serialized `.ji` file
baremodule Chemfiles_jll
using Base
using Base: UUID
import JLLWrappers

JLLWrappers.@generate_main_file_header("Chemfiles")
JLLWrappers.@generate_main_file("Chemfiles", UUID("78a364fa-1a3c-552a-b4bb-8fa0f9c1fcca"))
end  # module Chemfiles_jll
