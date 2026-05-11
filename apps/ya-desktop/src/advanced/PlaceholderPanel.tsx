export function PlaceholderPanel({
  title,
  description,
}: {
  title: string
  description: string
}) {
  return (
    <div className="p-6">
      <section className="rounded-3xl border border-slate-200 bg-white p-8 shadow-sm">
        <p className="text-sm font-medium text-blue-600">Advanced Runtime</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950">
          {title}
        </h1>
        <p className="mt-3 max-w-2xl text-sm leading-6 text-slate-500">
          {description}
        </p>
      </section>
    </div>
  )
}
