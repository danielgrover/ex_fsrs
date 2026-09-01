defmodule ExFsrs.MixProject do
  use Mix.Project

  @version "0.3.0"
  @source_url "https://github.com/danielgrover/ex_fsrs"
  @description "FSRS-6 spaced repetition scheduling, with an optional parameter optimizer."

  def project do
    [
      app: :ex_fsrs,
      version: @version,
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      elixirc_paths: elixirc_paths(Mix.env()),
      dialyzer: [plt_add_apps: [:ex_unit]],
      aliases: aliases(),
      test_coverage: [
        summary: [threshold: 90],
        # Needs the gated Anki dataset; exercised only by `mix test --include dataset`.
        ignore_modules: [ExFsrs.AnkiDataset]
      ],
      name: "ExFsrs",
      description: @description,
      source_url: @source_url,
      package: package(),
      docs: docs()
    ]
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => @source_url},
      files: ~w(lib mix.exs README.md LICENSE)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: ["README.md", "bench/NX_WHILE_GRAD_F64.md"],
      groups_for_modules: [
        Scheduling: [ExFsrs, ExFsrs.Scheduler, ExFsrs.ReviewLog],
        Optimizer: [
          ExFsrs.Optimizer,
          ExFsrs.Optimizer.Data,
          ExFsrs.Optimizer.Data.Review,
          ExFsrs.Optimizer.Initialization,
          ExFsrs.Optimizer.Metrics
        ],
        "Optimizer internals": [
          ExFsrs.Optimizer.Adam,
          ExFsrs.Optimizer.Loss,
          ExFsrs.Optimizer.Model,
          ExFsrs.Optimizer.Model.Batched,
          ExFsrs.Optimizer.Model.Loop
        ]
      ]
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_env), do: ["lib"]

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  def cli do
    [
      preferred_envs: [ci: :test]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # Optional: only needed for ExFsrs.Optimizer. Scheduling stays dependency-free.
      nx_dep(),
      {:exla, "~> 0.13", optional: true},
      {:ex_slop, "~> 0.4", only: [:dev, :test], runtime: false},
      {:reach, "~> 2.0", only: [:dev, :test], runtime: false},
      {:ex_dna, "~> 1.0", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.0", only: [:dev, :test], runtime: false},
      {:vibe_kit, "~> 0.1", only: [:dev, :test], runtime: false},
      {:credo, "~> 1.0", only: [:dev, :test], runtime: false},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end

  # Stock Nx runs the optimizer. Only `model: :loop`, which the bench scripts
  # use for speed, needs an Nx carrying the `while` f64 gradient fix described
  # in bench/NX_WHILE_GRAD_F64.md. Point EX_FSRS_NX_PATH at such a checkout
  # (e.g. ../nx/nx) to run them; otherwise `:loop` refuses to run and the
  # default `:batched` model is used.
  defp nx_dep do
    case System.get_env("EX_FSRS_NX_PATH") do
      nil -> {:nx, "~> 0.13", optional: true}
      path -> {:nx, path: path, override: true, optional: true}
    end
  end

  defp aliases() do
    [
      ci: [
        "compile --warnings-as-errors",
        "format --check-formatted",
        "test",
        "credo --strict",
        "dialyzer",
        "ex_dna --max-clones 0",
        "reach.check --arch --smells"
      ]
    ]
  end
end
